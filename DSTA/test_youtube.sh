#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train.py \
    --img_dim 1024 \
    --word_dim 1024 \
    --embed_size 1024 \
    --n_caption 1 \
    --learning_rate 1e-3 \
    --grad_clip 2 \
    --batch_size 4 \
    --num_epochs 50 \
    --measure end2end_dp \
    --phase test \
    --margin 0.1 \
    --model-type v2 \
    --data-name youtube \
    --parallel-test \
    --resume runs/movie_v2_train_pooling_pathloss_dploss_lambda0.1_floatfix_lr1e-3/checkpoint_24.pth.tar \
    --dump-name runs/movie_v2_train_pooling_pathloss_dploss_lambda0.1_floatfix_lr1e-3/test_embs
