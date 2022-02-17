#!/bin/bash
cmd="CUDA_VISIBLE_DEVICES=0 python train.py
    --expid tophead
    --model_name resnet18
    --dataset_parts_root /data/fcl_data/initial_CAD
    --num_class 2
    --batch_size 32
    --face_part_names 4
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
    "

echo ${cmd}
eval ${cmd}
