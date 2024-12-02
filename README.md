# Requirements
python, pytorch, timm, wandb, einops

# Data preparation

Downloads datasets and converts them to df (train/test.csv with class_id/dir format):

```
python make_df_from_dataset.py --dataset_name cub --dataset_root_path data/cub
```

# Train
Pytorch tutorial on CIFAR10 with LeNet:
```
python run_blitz.py
```

Modified version to use arguments to change model, image size and other settings:
```
python run_blitz_functional.py
```

Expanded to use models from timm, more data augmentation and regularization and 
custom datasets:
```
python run.py --cfg configs/cub.yaml
```
