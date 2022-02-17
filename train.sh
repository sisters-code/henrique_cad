#!/bin/bash
cmd="CUDA_VISIBLE_DEVICES=0 python train.py
    --expid hospi_NoEar_ADDaffine
    --split data/hospital/
    --model_name resnet18
    --num_class 2
    --batch_size 32
    --face_part_names no_ear
    --num_parts 1
    --gpus 1
    --log_dir multi_parts_logs
    --lr 0.001
    --wd 0.0
    --max_epochs 35
    --multistep 15 25
    --reduce_fc_channels 0
    --balance_samples
    --use_sam
    --use_swa
    --freeze_bn
    --square_resize
    --write_predictions
    "
#    --balance_samples
#2class_randHorFlip
#     --aug_train_scales 1.0
#     right_ear left_ear
#    --use_sam
#    --use_swa
#    --freeze_bn
echo ${cmd}
eval ${cmd}
