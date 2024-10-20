# Training Cutie

## Setting Up Data

We prepare datasets in the following structure:

```bash
├── LiVOS (codebase)
├── DAVIS
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── LVOS
│   ├── valid
|   |   ├──Annotations
|   |   └── ...
|   └── test
|       ├──Annotations
|       └── ...
├── YouTube_2019
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   └── valid
└── MOSE
    ├── JPEGImages
    └── Annotations
```

Links to the datasets:
- DAVIS (2017): https://davischallenge.org/
- YouTubeVOS (2019): https://youtube-vos.org/
- MOSE: https://henghuiding.github.io/MOSE/
- LVOS (v1): https://lingyihongfd.github.io/lvos.github.io/

## Training Command

We trained with four A100 GPUs, which took around 30 hours.

```
OMP_NUM_THREADS=4 torchrun --master_port 25357 --nproc_per_node=4 cutie/train.py exp_id=[some unique id] model=[small/base] data=[base/with-mose/mega]
```

- Change `nproc_per_node` to change the number of GPUs.
- Prepend `CUDA_VISIBLE_DEVICES=...` if you want to use specific GPUs.
- Change `master_port` if you encounter port collision.
- `exp_id` is a unique experiment identifier that does not affect how the training is done.
- Models and visualizations will be saved in `./output/`.
- For pre-training only, specify `main_training.enabled=False`.
- For main training only, specify `pre_training.enabled=False`.
- To load a pre-trained model, e.g., to continue main training from the final model from pre-training, specify `weights=[path to the model]`.