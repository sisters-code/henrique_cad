#!/bin/bash
cmd="CUDA_VISIBLE_DEVICES=1 python lime_visualize.py
    --expid lime_train
    --split data/train_splits_ray/
    --model_name resnet18
    --num_class 2
    --batch_size 32
    --face_part_names ONLY_FACE
    --num_parts 1
    --gpus 1
    --log_dir multi_parts_logs
    --lr 0.001
    --wd 0.0
    --max_epochs 35
    --multistep 15 25
    --reduce_fc_channels 0
    --use_sam
    --use_swa
    --freeze_bn
    --square_resize
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