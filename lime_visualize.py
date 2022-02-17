import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import timm
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import timm
import torch

from model import ModelMultiPart
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import FaceDataset4Crops, MultiFacePartDataset
import cv2
import csv
import codecs
from lime.wrappers.scikit_image import SegmentationAlgorithm

# # model = models.resnet18(pretrained=False)
# layer_list = list(timm.create_model('resnet18', pretrained=True, in_chans=3).children())[:-1]
# out_fc = nn.Sequential(nn.Dropout(0), nn.Linear(512, 2))
# layer_list.append(out_fc)
# backbones = nn.ModuleList([nn.Sequential(*layer_list)])

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

def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                    std=[0.225, 0.224, 0.229])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

def normalize(map):
    rescaled_map = map - np.amin(map)
    rescaled_map = rescaled_map / np.amax(map)
    return rescaled_map


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def main(args):
    model = ModelMultiPart(args)
    # weights_path = '/home/aa/code/henrique_cad/multi_parts_logs/resnet18/baseline_num_parts_1_16_256_0.001_balance_True_swa_True_sam_True_multi_step_15_25/version_1/checkpoints/resnet18_best_auroc=0.73_epoch=5_step=3023.ckpt'
    weights_path = '/home/aa/code/henrique_cad/multi_parts_logs/resnet18/face_cutout_num_parts_1_32_256_0.001_balance_True_swa_True_sam_True_multi_step_15_25/version_6/checkpoints/resnet18_best_auroc=0.73_epoch=7_step=2015.ckpt'
    model = model.load_from_checkpoint(weights_path)
    model.eval()

    norm_mean = (0.406, 0.456, 0.485)
    norm_std = (0.225, 0.224, 0.229)

    if args.square_resize:
        resize = (256,256)
        crop_size = (224,224)

    transform = T.Compose([
        T.Resize(tuple(resize)),
        T.CenterCrop(tuple(crop_size)),
        T.Normalize(norm_mean, norm_std)
    ])
    train_dataset = MultiFacePartDataset(
        args.dataset_parts_root,
        split_file=Path(args.split + 'train.txt'),
        transform=transform, balance_samples=args.balance_samples,
        part_names=args.face_part_names,
        use_gray=args.use_gray,
        use_eqhist=args.use_eqhist,
        use_gamma=args.use_gamma
    )
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=False, num_workers=16, pin_memory=False,
        drop_last=False
    )

    valid_dataset = MultiFacePartDataset(
        args.dataset_parts_root,
        split_file=Path(args.split + 'val.txt'),
        transform=transform,
        part_names=args.face_part_names,
        use_gray=args.use_gray,
        use_eqhist=args.use_eqhist,
        use_gamma=args.use_gamma
    )
    valid_dataloader = DataLoader(
        valid_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=False,
        drop_last=False
    )

    preprocess_transform = get_preprocess_transform()
    explainer = lime_image.LimeImageExplainer()

    def batch_predict(images):
        model.eval()

        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    n = len(train_dataset)
    # mask_saved_path = '/data/fcl_data/' + args.face_part_names[0] +'_lime_train/'
    mask_saved_path = '/data/fcl_data/' + args.face_part_names[0] + '_lime_valid/'
    os.makedirs(mask_saved_path, exist_ok=True)
    pill_transf = get_pil_transform()
    pos_map = np.zeros((224,224))
    neg_map = np.zeros((224,224))
    mask_list = np.zeros((n ,224,224))
    g_step = 0
    # for i, input in enumerate(train_dataloader):
    for i, input in enumerate(valid_dataloader):
        # logits = model(input)
        # probs = F.softmax(logits, dim=1)

        for path in input['path']:
            img = get_image(path)

            explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                                     batch_predict,  # classification function
                                                     top_labels=2,
                                                     hide_color=0,
                                                     num_samples=1000)  # number of images that will be sent to classification function
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                        hide_rest=False)
            img_boundry2 = mark_boundaries(temp / 255.0, mask)

            mask_list[g_step, :, :] = mask
            g_step += 1
            pos_map[np.where(mask == 1)] += 1
            neg_map[np.where(mask == -1)] += 1

            img_name = Path(path).parent.name
            img_path = mask_saved_path + img_name + '.jpg'
            # plt.imsave( img_path, img_boundry2)
    np.save(args.face_part_names[0] +'valid_mask_list', mask_list)
    # np.save(args.face_part_names[0] +'train_mask_list', mask_list)
    pos_map = normalize(pos_map)
    neg_map = normalize(neg_map)
    pos_heatmap = cv2.applyColorMap(np.uint8(255 * pos_map), cv2.COLORMAP_JET)
    neg_heatmap = cv2.applyColorMap(np.uint8(255 * neg_map), cv2.COLORMAP_JET)
    # plt.imsave( args.face_part_names[0] +'train_pos_map.jpg', pos_heatmap)
    # plt.imsave( args.face_part_names[0] +'train_neg_map.jpg', neg_heatmap)
    plt.imsave( args.face_part_names[0] +'valid_pos_map.jpg', pos_heatmap)
    plt.imsave( args.face_part_names[0] +'valid_neg_map.jpg', neg_heatmap)

if __name__ == '__main__':
    parser = _init_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)


