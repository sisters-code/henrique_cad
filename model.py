from argparse import Namespace
from pathlib import Path
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchmetrics.utilities.checks import _input_format_classification
from collections import Counter
import os

from dataset import FaceDataset4Crops, MultiFacePartDataset, TopHeadDataset
from sam import SAM
from utils import deleteDuplicatedElementFromList

SIZE_MULT = {
    'both_cheeks': (1, 1.68),
    'face': (1.33, 1),
    'head_front': (1.33, 1),
    'forehead': (1, 2.65),
    'left_cheek': (1.42, 1),
    'left_eye': (1, 2.33),
    'left_eye_surround': (1, 1.5),
    'nose': (1.15, 1),
    'right_cheek': (1.42, 1),
    'right_eye': (1, 2.33),
    'right_eye_surround': (1, 1.5),
    'left_ear': (1.28, 1),
    'head_left': (1.33, 1),
    'right_ear': (1.28, 1),
    'head_right': (1.33, 1)
}

def get_size(part_name, base_size, stride):
    size = (np.array(list(SIZE_MULT[part_name])) * base_size).astype(np.int32)
    size = size // stride * stride
    return size


class ModelMultiPart(pl.LightningModule):
    def __init__(
        self,
        args: Namespace,
        optuna_trial = None
    ) -> None:
        super().__init__()
        self.epoch = 0

        with open('/home/aa/code/henrique_cad/data/train_splits_multiclass/test_3cls.txt', 'r') as f:
            lines = f.read().strip().splitlines()
        lines = [ln.split(' ') for ln in lines]
        self.names = [ln[0].split('/')[1] for ln in lines]
        self.labels = [int(ln[1]) for ln in lines]

        log_model_dir = str(Path(args.log_dir) / args.model_name)

        self.pred_dir = os.path.join(log_model_dir, args.expid + '_num_parts_' + str(args.num_parts) + '_' + str(args.batch_size) + '_' + str(args.crop_size[0]) + \
                        '_' + str(args.lr) + '_balance_' + str(args.balance_samples) + '_swa_' + str(args.use_swa) + '_sam_' + \
                        str(args.use_sam) + '_multi_step_' + str(args.multistep[0]) + '_' + str(args.multistep[1]) )

        self.args = args
        self.optuna_trial = optuna_trial

        if args.num_parts is None:
            args.num_parts = len(args.face_part_names)

        in_chans = 3
        if self.args.use_gray:
            in_chans = 1
        # self.backbones = nn.ModuleList([nn.Sequential(*list(timm.create_model(args.model_name, pretrained=True, in_chans=in_chans).children())[:-1]) for _ in range(args.num_parts)])
        layer_list = list(timm.create_model(args.model_name, pretrained=True, in_chans=in_chans).children())[:-1]
        if self.args.expid[-4:] == 'rn10':
            layer_list.pop(-2)
            layer_list.pop(-2)
        self.backbones = nn.ModuleList([nn.Sequential(*layer_list) for _ in range(args.num_parts)])

        if args.model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            self.norm_mean = (0.406, 0.456, 0.485)
            self.norm_std = (0.225, 0.224, 0.229)
            if self.args.use_gray:
                self.norm_mean = 0.45
                self.norm_std = 0.225
        else:
            self.norm_mean = 0.5
            self.norm_std = 0.5
        if args.model_name in ['resnet18', 'resnet34']:
            ch = 512
            self.resize = 256
            self.crop_size = 224
            self.crop_stride = 32
        elif args.model_name in ['resnet50', 'resnet101']:
            ch = 2048
            self.resize = 256
            self.crop_size = 224
            self.crop_stride = 32
        elif args.model_name in ['tf_efficientnetv2_s', 'tf_efficientnetv2_m', 'tf_efficientnetv2_l']:
            ch = 1280
            self.head_layers = ['.classifier.']
            self.resize = 350
            self.crop_size = 300
            self.crop_stride = 10
        elif args.model_name.startswith('tf_efficientnet_'):
            self.head_layers = ['.classifier.']
            if 'b0' in args.model_name:
                ch = 1280
                self.resize = 256
                self.crop_size = 224
                self.crop_stride = 32
            elif 'b2' in args.model_name:
                ch = 1408
                self.resize = 300
                self.crop_size = 260
                self.crop_stride = 10
            elif 'b4' in args.model_name:
                ch = 1792
                self.resize = 430
                self.crop_size = 380
                self.crop_stride = 10
            elif 'b6' in args.model_name:
                ch = 2304
                self.resize = 580
                self.crop_size = 528
                self.crop_stride = 16

        if self.args.expid[-4:] == 'rn10':
            ch = 128

        self.reduces = None
        if self.args.reduce_fc_channels > 0:
            self.reduces = nn.ModuleList([nn.Sequential(nn.Dropout(args.dropout), nn.Linear(ch, self.args.reduce_fc_channels), nn.ReLU(True)) for _ in range(args.num_parts)])
            ch = self.args.reduce_fc_channels

        self.out_fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(len(args.face_part_names)*ch, args.num_class))

        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_noreduce = nn.CrossEntropyLoss(reduction='none')

        self.save_hyperparameters()

        self.counter = 0
        self.val_auroc_hist = []

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        images = {k: v for k, v in inputs.items() if k in self.args.face_part_names}
        if self.args.num_parts == 1:
            for v in images.values():
                feats = [
                    self.backbones[0](v)
                ]
            if self.args.reduce_fc_channels > 0:
                feats = [
                    self.reduces[0](feats[0])
                ]
        elif self.args.num_parts == 2:
            feats = [
                self.backbones[0](images['face']),
                self.backbones[1](images['left_ear']),
                self.backbones[1](images['right_ear'])
            ]

            feats = [
                self.reduces[0](feats[0]),
                self.reduces[1](feats[1]),
                self.reduces[1](feats[2])
            ]

        x = torch.cat(feats, dim=1)
        logits = self.out_fc(x)
        return logits
    # def forward(
    #     self,
    #     image
    # ):
    #     feats = [
    #         self.backbones[0](image)
    #     ]
    #     x = torch.cat(feats, dim=1)
    #     logits = self.out_fc(x)
    #     return logits



    def configure_optimizers(self):
        if self.args.freeze_bn:
            self.freeze_bn()

        if self.args.use_sam:
            self.automatic_optimization = False

        base_optim = optim.SGD
        optim_kwargs = {'momentum': 0.9}
        # base_optim = optim.AdamW
        # optim_kwargs = {}

        if self.args.backbone_lr_multiplier != 1:
            backbone_params = []
            head_params = []
            head_names = []
            for n, p in self.named_parameters():
                if not 'shadow' in n:
                    is_in = [h in n for h in self.head_layers]
                    if any(is_in):
                        head_names.append(n)
                        head_params.append(p)
                    else:
                        backbone_params.append(p)

            if self.args.backbone_lr_multiplier < 1e-7:
                for n, p in self.named_parameters():
                    if n not in head_names:
                        p.requires_grad_ = False
                if self.args.use_sam:
                    opt = SAM(
                        head_params, base_optim, lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs
                    )
                else:
                    opt = base_optim(
                        head_params, lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs)
            else:
                if self.args.use_sam:
                    opt = SAM(
                        [
                            {'params': head_params},
                            {'params': backbone_params, 'lr': self.args.lr*self.args.backbone_lr_multiplier}
                        ], base_optim,
                        lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs
                    )
                else:
                    opt = base_optim(
                        [
                            {'params': head_params},
                            {'params': backbone_params, 'lr': self.args.lr*self.args.backbone_lr_multiplier}
                        ],
                        lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs)
        else:
            if self.args.use_sam:
                opt = SAM(
                    self.parameters(), base_optim, lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs
                )
            else:
                opt = base_optim(
                    self.parameters(), lr=self.args.lr, weight_decay=self.args.wd, **optim_kwargs)


        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, self.args.multistep, self.args.step_gamma)
        # lr_scheduler = optim.lr_scheduler.OneCycleLR(
        #     opt, max_lr=self.args.lr, epochs=self.args.max_epochs, steps_per_epoch=248, cycle_momentum=False)

        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch'}}

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.args.freeze_epoch:
            for n, p in self.named_parameters():
                if not (n.startswith('backbone.fc') or n.startswith('backbone.layer4')):
                    p.requires_grad_(False)

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        if len(self.args.aug_train_scales) > 0:
            i = random.randint(0, len(self.args.aug_train_scales)-1)
            for k in self.args.face_part_names:
                batch[k] = F.interpolate(batch[k], scale_factor=self.args.aug_train_scales[i], mode='bilinear', align_corners=True)
        preds = self(batch)
        loss = self.loss_fn(preds, batch['label'])
        self.log('train/loss', loss, on_step=True, prog_bar=True)

        if self.args.use_sam:
            optimizer = self.optimizers()

            # first forward-backward pass
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            preds = self(batch)
            loss2 = self.loss_fn(preds, batch['label'])
            self.manual_backward(loss2)
            optimizer.second_step(zero_grad=True)

        return {
            'loss': loss,
            'preds': preds.detach().cpu(),
            'targets': batch['label'].detach().cpu(),
            'paths': batch['path']
        }

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        preds = torch.cat([o['preds'] for o in outputs], dim=0)
        preds = torch.softmax(preds, dim=1)
        targets = torch.cat([o['targets'] for o in outputs], dim=0)
        try:
            auroc = torchmetrics.functional.auroc(preds, targets, self.args.num_class)
            hit_acc = torchmetrics.functional.accuracy(preds, targets, num_classes=self.args.num_class)
            confuse_matirx = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=self.args.num_class)
            confuse_matirx = confuse_matirx.numpy()
            TP, FP, TN, FN = confuse_matirx[1][1], confuse_matirx[0][1], confuse_matirx[0][0], confuse_matirx[1][0]
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
        except ValueError:
            auroc = 0
            hit_acc = 0
            recall = 0
            precision = 0
        self.log('train/auroc', auroc, prog_bar=False)
        self.log('train/acc', hit_acc, prog_bar=False)
        self.log('train/recall', recall, prog_bar=False)
        self.log('train/precision', precision, prog_bar=False)

        # explanation = self.explainer.explain_instance(np.array(pill_transf(img)),
        #                                          preds,  # classification function
        #                                          top_labels=2,
        #                                          hide_color=0,
        #                                          num_samples=2)
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2,
        #                                             hide_rest=False)
        # img_boundry1 = mark_boundaries(temp / 255.0, mask)
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=2,
        #                                             hide_rest=False)
        # img_boundry2 = mark_boundaries(temp / 255.0, mask)

        if self.args.write_predictions:
            # preds = torch.softmax(preds, dim=1)
            paths = []
            for o in outputs:
                paths.extend(o['paths'])
            paths = ['/'.join(p.split('/')[-2:]) for p in paths]

            targets = list(targets.numpy())
            preds = list(preds.numpy())
            paths, preds, targets = deleteDuplicatedElementFromList(paths, preds, targets)
            preds_0 = []
            preds_1 = []
            for i in range(len(preds)):
                preds_0.append(preds[i][0])
                preds_1.append(preds[i][1])
            preds_csv, targets_csv, mode = _input_format_classification(torch.tensor(np.array(preds)), torch.tensor(targets), 0.5)
            df = pd.DataFrame({
                'id': paths,
                'label': targets,
                'preds_0': preds_0,
                'preds_1': preds_1,
                'result': (preds_csv == targets_csv)[:,0]
            })

            df.to_csv(Path(self.pred_dir) / ('train_epoch{}_'.format(self.epoch) +'preds.csv'), index=False)
            self.epoch += 1


    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> torch.Tensor:
        if self.args.use_tta:
            preds = self._pred_tta(batch)
        else:
            preds = self(batch)
        loss = self.loss_fn(preds, batch['label'])
        
        loss_batch = self.loss_fn_noreduce(preds, batch['label'])
        # pred_idx = np.argmax(preds.detach().cpu().numpy(), axis=1)
        # for i in range(preds.shape[0]):
        #     im = (255*batch['image'][i].permute(1, 2, 0).detach().cpu().numpy()).astype(np.uint8).copy()
        #     cv.putText(im, f'target: {batch["label"][i]}', (10, 25), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        #     cv.putText(im, f'pred: {pred_idx[i]}', (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        #     cv.putText(im, f'loss: {loss_batch[i]:.03f}', (10, 75), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        #     cv.imwrite(f'outputs/val_{self.counter}.jpg', im)
        #     self.counter += 1

        self.log('val/loss', loss)
        return {
            'preds': preds.detach().cpu(),
            'targets': batch['label'].detach().cpu(),
            'losses': loss_batch.detach().cpu(),
            'paths': batch['path']
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        preds = torch.cat([o['preds'] for o in outputs], dim=0)
        preds = torch.softmax(preds, dim=1)
        targets = torch.cat([o['targets'] for o in outputs], dim=0)
        try:
            # auroc = torchmetrics.functional.auroc(preds, targets, 2)
            auroc = torchmetrics.functional.auroc(preds, targets, num_classes=self.args.num_class)
            hit_acc = torchmetrics.functional.accuracy(preds, targets, num_classes=self.args.num_class)
            confuse_matirx = torchmetrics.functional.confusion_matrix(preds, targets, num_classes=self.args.num_class)
            confuse_matirx = confuse_matirx.numpy()
            TP, FP, TN, FN = confuse_matirx[1][1], confuse_matirx[0][1], confuse_matirx[0][0], confuse_matirx[1][0]
            recall = TP / (TP+FN)
            precision = TP/ (TP+FP)
            # recall = torchmetrics.functional.recall(preds, targets, top_k=1, num_classes=self.args.num_class)
            # precision = torchmetrics.functional.precision(preds, targets, num_classes=self.args.num_class)
        except ValueError:
            auroc = torch.zeros(1).to(device=preds.device, dtype=preds.dtype)
            hit_acc = torch.zeros(1).to(device=preds.device, dtype=preds.dtype)
            recall = 0
            precision = 0

        self.log('auroc', auroc, prog_bar=False)
        self.log('val/auroc', auroc, prog_bar=False)
        self.log('val/acc', hit_acc, prog_bar=False)
        self.log('val/recall', recall, prog_bar=False)
        self.log('val/precision', precision, prog_bar=False)

        '''
        count = Counter(self.labels)
        total_0 =count[0]
        total_1 = count[1]
        total_2 = count[2]

        # self.val_auroc_hist.append(auroc.item())
        preds, target, mode = _input_format_classification(preds, targets, 0.5)
        p_class0,p_class1,p_class2 = 0,0,0
        n_class0, n_class1, n_class2 = 0,0,0
        paths = []
        for o in outputs:
            paths.extend(o['paths'])
        for index,item in enumerate(preds.numpy().tolist()):
            if item == [0,1]:
                i = self.names.index(Path(paths[index]).parent.name)
                if self.labels[i] == 0:
                    p_class0 += 1
                elif self.labels[i] == 1:
                    p_class1 += 1
                elif self.labels[i] == 2:
                    p_class2 += 1
            else:
                i = self.names.index(Path(paths[index]).parent.name)
                if self.labels[i] == 0:
                    n_class0 += 1
                elif self.labels[i] == 1:
                    n_class1 += 1
                elif self.labels[i] == 2:
                    n_class2 += 1

        self.log('ratio/p_class0', p_class0/total_0, prog_bar=False)
        self.log('ratio/p_class1', p_class1/total_1, prog_bar = False)
        self.log('ratio/p_class2', p_class2/total_2, prog_bar = False)
        self.log('ratio/n_class0', n_class0/total_0, prog_bar = False)
        self.log('ratio/n_class1', n_class1/total_1, prog_bar = False)
        self.log('ratio/n_class2', n_class2/total_2, prog_bar = False)
        # preds = preds.argmax(dim=1)
        # target = target.argmax(dim=1)
        '''
        if self.args.write_predictions:
            # preds = torch.softmax(preds, dim=1)
            paths = []
            for o in outputs:
                paths.extend(o['paths'])
            paths = ['/'.join(p.split('/')[-2:]) for p in paths]
            preds_csv, targets_csv, mode = _input_format_classification(preds, targets, 0.5)
            df = pd.DataFrame({
                'id': paths,
                'label': list(targets.numpy()),
                'preds_0': list(preds[:, 0].numpy()),
                'preds_1': list(preds[:, 1].numpy()),
                'result': (preds_csv == targets_csv)[:,0]
            })

            df.to_csv(Path(self.pred_dir) / ('valid_epoch{}_'.format(self.epoch) +'preds.csv'), index=False)


    def train_dataloader(self) -> DataLoader:
        if self.args.square_resize:
            resize = (256,256)
            crop_size = (224,224)
        else:
            resize = get_size(self.args.face_part_names[0], self.resize, 1)
            crop_size = get_size(self.args.face_part_names[0], self.crop_size, self.crop_stride)
        if self.args.face_part_names is None:
            # use original 4 images dataset
            transform = T.Compose([
                T.Resize(tuple(resize)),
                T.RandomRotation(30, fill=1),
                T.RandomCrop(tuple(crop_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.Normalize(self.norm_mean, self.norm_std)
            ])
            dataset = FaceDataset4Crops(
                self.args.dataset_4crops_root,
                split_file=Path('data/train_splits_ray/train.txt'),
                transform=transform, balance_samples=self.args.balance_samples,
                image_names=self.args.image_names)
        else:
            # transform = T.Compose([
            #     T.Resize(tuple(resize)),
            #     T.RandomCrop(tuple(crop_size)),
            #     T.RandomHorizontalFlip(p=0.5),
            #     T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            #     T.RandomAffine(degrees=0, translate=(0, 0.2), scale=(0.8, 1),
            #                    interpolation=InterpolationMode.BILINEAR),
            #     T.RandomPerspective(),
            #     T.Normalize(self.norm_mean, self.norm_std)
            # ])
            transform = T.Compose([
                T.Resize(tuple(resize)),
                T.RandomCrop(tuple(crop_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1),
                               interpolation=InterpolationMode.BILINEAR),
                T.Normalize(self.norm_mean, self.norm_std)
            ])
            # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            # T.RandomAffine(degrees=0, translate=(0, 0.2), scale=(0.8, 1),
                           # interpolation=InterpolationMode.BILINEAR),
            # T.RandomPerspective(),
            if self.args.use_gray:
                transform = T.Compose([
                    T.Resize(tuple(resize)),
                    T.RandomCrop(tuple(crop_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(self.norm_mean, self.norm_std)
                ])
            dataset = MultiFacePartDataset(
                self.args.dataset_parts_root,
                split_file=Path(self.args.split + 'train.txt'),
                transform=transform, balance_samples=self.args.balance_samples,
                part_names=self.args.face_part_names,
                use_gray=self.args.use_gray,
                use_eqhist=self.args.use_eqhist,
                use_gamma=self.args.use_gamma
            )
            # dataset = TopHeadDataset(
            #     self.args.dataset_parts_root,
            #     split_file=Path('data/train_splits_4crops/train.txt'),
            #     transform=transform, balance_samples=self.args.balance_samples,
            #     part_names=self.args.face_part_names,
            #     use_gray=self.args.use_gray,
            #     use_eqhist=self.args.use_eqhist,
            #     use_gamma=self.args.use_gamma
            # )
        loader = DataLoader(
            dataset, self.args.batch_size, shuffle=True, num_workers=16, pin_memory=False,
            drop_last=False
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        if self.args.square_resize:
            resize = (256,256)
            crop_size = (224,224)
        else:
            resize = get_size(self.args.face_part_names[0], self.resize, 1)
            crop_size = get_size(self.args.face_part_names[0], self.crop_size, self.crop_stride)
        if self.args.face_part_names is None:
            # use original 4 images dataset
            transform = T.Compose([
                T.Resize(tuple(resize)),
                T.CenterCrop(tuple(crop_size)),
                T.Normalize(self.norm_mean, self.norm_std)
            ])
            dataset = FaceDataset4Crops(
                self.args.dataset_4crops_root,
                split_file=Path(self.args.split + 'val.txt'),
                transform=transform,
                image_names=self.args.image_names
            )
        else:
            transform = T.Compose([
                T.Resize(tuple(resize)),
                T.CenterCrop(tuple(crop_size)),
                T.Normalize(self.norm_mean, self.norm_std)
            ])
            dataset = MultiFacePartDataset(
                self.args.dataset_parts_root,
                split_file=Path(self.args.split + 'val.txt'),
                transform=transform,
                part_names=self.args.face_part_names,
                use_gray=self.args.use_gray,
                use_eqhist=self.args.use_eqhist,
                use_gamma=self.args.use_gamma
            )
            # dataset = TopHeadDataset(
            #     self.args.dataset_parts_root,
            #     split_file=Path('data/train_splits_4crops/test.txt'),
            #     transform=transform,
            #     part_names=self.args.face_part_names,
            #     use_gray=self.args.use_gray,
            #     use_eqhist=self.args.use_eqhist,
            #     use_gamma=self.args.use_gamma
            # )

        return DataLoader(
            dataset, self.args.batch_size, shuffle=False, num_workers=4, pin_memory=False,
            drop_last=False
        )

    def _pred_tta(
        self,
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        orig_batch = {}
        for k in self.args.face_part_names:
            orig_batch[k] = batch[k]
        preds = None
        for s in self.args.tta_scales:
            for k in self.args.face_part_names:
                batch[k] = F.interpolate(orig_batch[k], scale_factor=s, mode='bilinear', align_corners=True)
            if preds is None:
                preds = self(batch)
            else:
                preds += self(batch)

            for k in self.args.face_part_names:
                batch[k] = torch.flip(batch[k], dims=[-1])
            preds += self(batch)

        return preds / (2 * len(self.args.tta_scales))
