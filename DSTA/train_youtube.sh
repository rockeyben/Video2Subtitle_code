#!/bin/bash
CUDA_VISIBLE_DEVICES=4  python train.py \
    --img_dim 1024 \
    --txt_dim 1024 \
    --embed_size 1024 \
    --n_caption 1 \
    --learning_rate 1e-3 \
    --grad_clip 2 \
    --batch_size 4 \
    --num_epochs 40 \
    --phase train \
    --margin 0.1 \
    --log_step 1000 \
    --data-name youtube \
    --data-path /S4/MI/xueb/data/Video2Subtitle \
    --lbd_ord 1 \
    --lbd_dur 5 \
    --lbd_len 0.2 \
    --logger_name runs/check_youtube