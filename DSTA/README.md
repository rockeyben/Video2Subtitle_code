# DSTA module

The main core is in `end2end_dp_batch.py`.

## Environment

For train and inference, we use:
```
python=2.7.13
torch=0.4.1
```

For `match_movie.sh` and `match_youtube.sh`, we use:
```
python=3.6.13
ray=1.8.0
```



## Train

Run the following scripts after configuring:

- data path: `--data-path`
- logger path: `--logger_name`
```
./train_movie.sh
./train_youtube.sh
```

## Test

Run the following scripts to get intermediate embeddings after configuring:

- data path: `--data-path`
- model path: `--resume`
- dumped embeddings: `--dump-name`

```
./test_movie.sh
./test_youtube.sh
```

Run the following scripts to get final scores after configuring:

- dumped embeddings: `--test-path`
- dumped scores: `--save-path`

```
./match_movie.sh
./match_youtube.sh
```
