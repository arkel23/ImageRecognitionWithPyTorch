# Requirements
python, pytorch, timm, wandb, einops

```
pip install --no-dependencies --force-reinstall -v "timm==0.9.12
pip install -r requirements.txt
```

# Data preparation

Downloads datasets and converts them to df (train/test.csv with class_id/dir format):

```
mkdir data
cd data
wget -O ASDDataset.tar.gz https://huggingface.co/datasets/NYCU-PCSxNTHU-MIS/ABIDEI/resolve/main/ASDDataset.tar.gz?download=true
tar -xzvf ASDDataset.tar.gz
cd ..

# other datasets
# python make_df_from_dataset.py --dataset_name cub --dataset_root_path data/cub
```

# Train

Pytorch tutorial on CIFAR10 with LeNet modified version to use arguments to change model, image size and other settings:

```
python run_blitz_functional.py
```

Expanded to use models from timm, more data augmentation and regularization and 
custom datasets:

```
python run.py
python run.py --cfg configs/abide.yaml
```

# Inference:

Single image (can use a folder instead of image to inference over a folder):

```
python inference.py --cfg configs/abide.yaml --ckpt_path results/abide_resnet18_0/last.pth --images_path data/ASDDataset/ASD/51160.jpg --vis_mask GradCAM
```

Demo with Gradio (need to adjust datasets, ckpts for other cases):

```
python demo.py
```