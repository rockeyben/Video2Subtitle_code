#!/bin/bash

python demo.py \
    --data-name youtube \
    --data-path /S4/MI/xueb/data/Video2Subtitle \
    --dsta DSTA/runs/check_youtube/score.npy \
    --dsta-embs DSTA/runs/check_youtube/test_embs \
    --object GSM/score/gsm_youtube_object.npy \
    --scene GSM/score/gsm_youtube_scene.npy \
    --action GSM/score/gsm_youtube_action.npy