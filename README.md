# MOST
Official implementation of MOST: MR reconstruction Optimization for multiple downstream Tasks via continual learning
> H. Jeong, SY. Chun, and J. Lee, _MOST: MR reconstruction Optimization for multiple downstream Tasks via continual learning_, MICCAI 2025,
> [[arXiv]](https://arxiv.org/abs/2409.10394)

## Dependencies

Use `environment.yml` for required packages, or create a Conda environment with all dependencies:
```bash
conda env create -f environment.yml
```

## Dataset

The processed pickle data can be downloaded from google drive link in https://drive.google.com/drive/folders/1AuCYGiNOZ8hWrqiV_npsjmcodNVfRb6z?usp=share_link

## Pretrained models

The checkpoints can be downloaded from google drive link in https://drive.google.com/drive/folders/1AuCYGiNOZ8hWrqiV_npsjmcodNVfRb6z?usp=share_link


### MOST downstream task-oriented finetuning

Use `train_MOST.sh`.

### Downstream task evaluation for sequence of finetuning

Use `test_MOST.sh`.