#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --visual_feat_dim 1024 \
    --word_dim 300 \
    --embed_size 2048 \
    --n_caption 1 \
    --batch_size 128 \
    --visual_norm \
    --text_norm \
    --num_epochs 50 \
    --data-name youtube \
    --data-path /S4/MI/xueb/data/Video2Subtitle \
    --logger_name runs/youtube_action \
    --vis_input_type action
