from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import timm
import torch

from model import ModelMultiPart
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from dataset import MultiFacePartDataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'

def _init_parser():
    parser = ArgumentParser()
    parser.add_argument('--expid', type=str, default='debug')
    # parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--split', type=str, default='data/hospital/')
    parser.add_argument('--dataset_4crops_root', type=str, default='/data/fcl_data/face_CAD_data/')
    parser.add_argument('--dataset_parts_root', type=str, default='/data/fcl_data/proc_cad')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=timm.list_models())
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--square_resize', action='store_true')
    parser.add_argument('--crop_size', type=int, nargs=2, default=(256, 256))
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--multistep', type=int, nargs='+', default=(50, 90,))
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--freeze_epoch', type=int, default=1000)
    parser.add_argument('--balance_samples', action='store_true')
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--backbone_lr_multiplier', type=float, default=1.0)
    parser.add_argument('--image_names', type=str, nargs='+', default=('1.JPG',))
    parser.add_argument('--use_sam', action='store_true')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--clear_trainer', action='store_true')
    parser.add_argument('--reduce_fc_channels', type=int, default=512)
    parser.add_argument('--write_predictions', action='store_true')
    parser.add_argument('--face_part_names', type=str, nargs='+', default=None)
    parser.add_argument('--num_parts', type=int, default=None)
    parser.add_argument('--aug_train_scales', type=float, nargs='*', default=[])
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('--tta_scales', type=float, nargs='*', default=[1.0])
    parser.add_argument('--use_gray', action='store_true')
    parser.add_argument('--use_eqhist', action='store_true')
    parser.add_argument('--use_gamma', action='store_true')
    return parser


def main(args):
    model = ModelMultiPart(args)
    model = model.train()

    if args.resume_from_checkpoint is not None and args.clear_trainer:
        ckpt = torch.load(args.resume_from_checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        args.resume_from_checkpoint = None

    if args.log_dir is None:
        args.log_dir = 'lightning_logs'
    log_model_dir = str(Path(args.log_dir) / args.model_name)

    # (''.join(i) for i in args.face_part_names)
    tb_logger = pl.loggers.TensorBoardLogger(log_model_dir, name=args.expid + '_num_parts_' + str(args.num_parts) + '_' + str(args.batch_size) + '_'
                                            + str(args.crop_size[0]) + '_' + str(args.lr) + '_balance_' + str(args.balance_samples) + '_swa_' + str(args.use_swa)
                                             + '_sam_' + str(args.use_sam) + '_multi_step_' + str(args.multistep[0]) + '_' + str(args.multistep[1]))

    model_ckpt_last = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model_name+'_last_{epoch}_{step}', save_weights_only=True)
    model_ckpt_train = pl.callbacks.model_checkpoint.ModelCheckpoint(filename=args.model_name+'_train_{epoch}_{step}')
    model_ckpt_best = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filename=args.model_name+'_best_{auroc:.2f}_{epoch}_{step}', save_weights_only=True,
        save_top_k=1, monitor='auroc', mode='max')

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, auto_select_gpus=True, stochastic_weight_avg=args.use_swa, callbacks=[model_ckpt_last, model_ckpt_train, model_ckpt_best])
    # trainer = pl.Trainer(gpus='1')

    resize = (256, 256)
    crop_size = (224, 224)
    norm_mean = (0.406, 0.456, 0.485)
    norm_std = (0.225, 0.224, 0.229)

    T_trans = T.Compose([
        T.Resize(tuple(resize)),
        T.RandomCrop(tuple(crop_size)),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomAffine(degrees=10, translate=(0.1, 0.1),
        #                interpolation=InterpolationMode.BILINEAR),
        T.Normalize(norm_mean, norm_std)
    ])
    train_dataset = MultiFacePartDataset(
        args.dataset_parts_root,
        split_file=Path(args.split + 'train.txt'),
        transform=T_trans, balance_samples=args.balance_samples,
        part_names=args.face_part_names,
        use_gray=args.use_gray,
        use_eqhist=args.use_eqhist,
        use_gamma=args.use_gamma
    )
    V_trans = T.Compose([
        T.Resize(tuple(resize)),
        T.CenterCrop(tuple(crop_size)),
        T.Normalize(norm_mean, norm_std)
    ])
    valid_dataset = MultiFacePartDataset(
        args.dataset_parts_root,
        split_file=Path(args.split + 'val.txt'),
        transform=V_trans,
        part_names=args.face_part_names,
        use_gray=args.use_gray,
        use_eqhist=args.use_eqhist,
        use_gamma=args.use_gamma
    )
    dataset = ConcatDataset([train_dataset, valid_dataset])
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, sampler=train_subsampler)
        validloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size, sampler=test_subsampler)
        trainer.fit(model, train_dataloader=trainloader, val_dataloaders=validloader)


if __name__ == '__main__':
    parser = _init_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
