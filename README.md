# Video2Subtitle

Implementaion code for "Video2Subtitle: Matching Weakly-Synchronized Sequences via Dynamic Temporal Alignment" (ICMR 2022)


<iframe src="https://github.com/rockeyben/Video2Subtitle_code/tree/main/assets/case1_o.m4v" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

<iframe src="https://github.com/rockeyben/Video2Subtitle_code/tree/main/assets/case1_g.m4v" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>

## Train & Test

Please go to `DSTA/` and `GSM/` folder to conduct their training procedure.

## Fusion & Subtitle re-generation demo

Requirements:
```
python=3.6.13
cv2=4.5.3
csv=1.0
json=2.0.9
h5py=3.1.0
torch
```

Run `demo.py` after getting score.npy for each cues (DSTA, object, scene, action), here is an example to get topK (K=3) matched subtitles for a given video (videoID=0), you may also change fusion weights as you want:

```
python demo.py \
    --data-name youtube \
    --data-path /path/to/Video2Subtitle \
    --dsta DSTA/runs/check_youtube/score.npy \
    --dsta-embs DSTA/runs/check_youtube/test_embs \
    --object GSM/score/gsm_youtube_object.npy \
    --scene GSM/score/gsm_youtube_scene.npy \
    --action GSM/score/gsm_youtube_action.npy \
    --w_d 10.0 \
    --w_o 1.0 \
    --w_s 1.0 \
    --w_a 1.0 \
    --test-id 0 \
    --K 3
``` 
 
## Data structure

```
Video2Subtitle/
    
    Movie/
        raw_video/
        lip/
        tts/
        feature_object/
        feature_scene/
        feature_action/
        feature_lip/
        feature_tts/
        gt_path/
        script.csv
        script_noblank.json
        speaking_score.hdf5
        dsta_train_ids.txt
        gsm_train_ids.txt
        test_ids.txt
        video_info.json
        
    Youtube/
        ...
```

- `raw_video/` : original videos
- `speaking_score.hdf5` : speaking score for each frame in raw videos
- `lip/` : trimmed face sequences after removing non-conversational frames
- `tts/` : trimmed text (TTS audio) sequences after removing non-conversational frames
- `gt_path` : ground truth alignment path for lip & tts
- `script.csv` : original subtitles, timestamp metric: ms.
- `script_noblank.json` : trimmed subtitles after removing non-conversational frames
- `dsta_train_ids.txt` : training video ids for DSTA
- `gsm_train_ids.txt` : training video ids for GSM
- `test_ids.txt` : testing video ids for fusion
- `video_info.json` : dict, (video_id, [video_name, FPS, duration])



