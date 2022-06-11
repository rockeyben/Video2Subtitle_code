#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_movie_vrnn_tglove.py \
    --img_dim 2208 \
    --word_dim 300 \
    --embed_size 1024 \
    --visual_norm \
    --text_norm \
    --n_caption 1 \
    --resume runs_v2/youtube_scene_vrnn_tglove_v2/model_best.pth.tar \
    --vis_input_type scene
