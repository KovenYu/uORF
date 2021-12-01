# Unsupervised Discovery of Object Radiance Fields
![teaser](teaser.gif)

This is a fork of the [original repository](https://github.com/KovenYu/uORF) with support for multiple GPUs.


- arXiv link: [https://arxiv.org/abs/2107.07905](https://arxiv.org/abs/2107.07905)
- Project website: [https://kovenyu.com/uorf](https://kovenyu.com/uorf)

## Environment
We recommend using Conda:
```sh
conda env create -f environment.yml
conda activate uorf
```
or install the packages listed therein. Please make sure you have NVIDIA drivers supporting CUDA 11.0, or modify the version specifictions in `environment.yml`.

## Data and model
Please download datasets and models [here](https://office365stanford-my.sharepoint.com/:f:/g/personal/koven_stanford_edu/Et9SOVcOxOdHilaqfq4Y3PsBsiPGW6NGdbMd2i3tRSB5Dg?e=WRrXIh).
If you want to train on your own dataset, please refer to this [README](data/README.md).

## Training
We assume you have at least one GPU with no less than 24GB memory (evaluation does not require this as rendering can be done ray-wise but some losses are defined on the image space),
e.g., 3090. Then run
```shell
bash custom/train_room_diverse.sh PATH_TO_DATASET
```
e.g.
```shell
bash scripts/train_clevr_567.sh ../data/1200shape_50bg
```
Training takes ~1 day on eight V100 for Room-Diverse.
During training, visualization will be sent to the `lightning-logs` directory and can be accessed with tensorboard.

If you want to specify the GPUs to use, add the `--gpus` flag to the training script.
The argument is parsed as string, see all possibilities in the pytorch-lightning [docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html#select-gpu-devices).

## Bibtex
```
@article{yu2021unsupervised
  author    = {Yu, Hong-Xing and Guibas, Leonidas J. and Wu, Jiajun},
  title     = {Unsupervised Discovery of Object Radiance Fields},
  journal   = {arXiv preprint arXiv:2107.07905},
  year      = {2021},
}
```
