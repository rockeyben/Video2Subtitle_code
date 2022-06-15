# GSM module

We use Dual-Encoding [[code]](https://github.com/danieljf24/dual_encoding) as our code base.

## Environment

For train and inference, we use:
```
python=2.7.13
torch=0.4.1
```

## Train

Run the following scripts after configuring:

- data path: `--data-path`
- logger path: `--logger_name`
```
./train_movie_object.sh
./train_movie_scene.sh
./train_movie_action.sh

./train_youtube_object.sh
./train_youtube_scene.sh
./train_youtube_action.sh
```

## Test

Run the following scripts after configuring:

- data path: `--data-path`
- model path: `--resume`
- output score npy path: `--save-path`


```
./test_movie_object.sh
./test_movie_scene.sh
./test_movie_action.sh

./test_youtube_object.sh
./test_youtube_scene.sh
./test_youtube_action.sh
```