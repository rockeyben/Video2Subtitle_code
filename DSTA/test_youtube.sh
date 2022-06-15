#!/bin/bash
CUDA_VISIBLE_DEVICES=3  python train.py \
    --img_dim 1024 \
    --txt_dim 1024 \
    --embed_size 1024 \
    --n_caption 1 \
    --batch_size 1 \
    --phase test \
    --data-path /S4/MI/xueb/data/Video2Subtitle \
    --data-name youtube \
    --lbd_len 0.2 \
    --parallel-test \
    --resume runs/check_youtube/model_best.pth.tar \
    --dump-name runs/check_youtube/test_embs
