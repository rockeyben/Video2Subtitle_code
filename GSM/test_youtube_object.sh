#!/bin/bash
data_path='/S4/MI/xueb/data/Video2Subtitle'
data_name='youtube'
data_type='object'
model_path='/S4/MI/xueb/data/Video2Subtitle/pretrain_models/GSM_youtube_object.pth.tar'
save_path='score/gsm_youtube_object.npy'

CUDA_VISIBLE_DEVICES=0 python test.py \
    --visual_feat_dim 2208 \
    --word_dim 300 \
    --embed_size 1024 \
    --batch_size 128 \
    --visual_norm \
    --text_norm \
    --n_caption 1 \
    --data-name ${data_name} \
    --data-path ${data_path} \
    --resume ${model_path} \
    --vis_input_type ${data_type} \
    --save-path ${save_path}
