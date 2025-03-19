import os
import time
import yaml
import math
import random
import argparse

import torch
from torch import nn

import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms

import timm
from timm.loss import LabelSmoothingCrossEntropy
import wandb
from einops.layers.torch import Reduce
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

from mix import get_mix, mixup_criterion


ImageFile.LOAD_TRUNCATED_IMAGES = True


def yaml_config_hook(config_file):
    '''
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    '''

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get('defaults', []):
            fp = cfg.get('defaults').get(d)
            cf = os.path.join(os.path.dirname(config_file), fp)
            with open(cf) as f:
                val = yaml.safe_load(f)
                print(val)
                cfg.update(val)

    if 'defaults' in cfg.keys():
        del cfg['defaults']

    return cfg


def parse_args():
    parser = argparse.ArgumentParser('Arguments for code')

    # training in general
    parser.add_argument('--serial', type=int, default=0, help='id for run')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='freq for logging train results')


    # dataset
    parser.add_argument('--dataset_name', default='cifar10', type=str,
                        help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default='data',
                        help='the root directory for where the data/feature/label files are')

    # folders with images (can be same: those where it's all stored in 'data')
    parser.add_argument('--folder_train', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/train/')
    parser.add_argument('--folder_val', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/val/')
    parser.add_argument('--folder_test', type=str, default='data',
                        help='the directory where images are stored, ex: dataset_root_path/test/')

    # df files with img_dir, class_id
    parser.add_argument('--df_train', type=str, default='train.csv',
                        help='the df csv with img_dirs, targets, def: train.csv')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv',
                        help='the df csv with img_dirs, targets, def: train_val.csv')
    parser.add_argument('--df_val', type=str, default='val.csv',
                        help='the df csv with img_dirs, targets, def: val.csv')
    parser.add_argument('--df_test', type=str, default='test.csv',
                        help='the df csv with img_dirs, targets, root/test.csv')
    parser.add_argument('--df_classid_classname', type=str, default='classid_classname.csv',
                        help='the df csv with classnames and class ids, root/classid_classname.csv')

    parser.add_argument('--train_trainval', action='store_false',
                        help='when true uses trainval for train and evaluates on test \
                        otherwise use train for train and evaluates on val')
    parser.add_argument('--cfg', type=str,
                        help='If using it overwrites args and reads yaml file in given path')


    # optimizer
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')


    # model
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true',
                        help='pretrained model on imagenet')
    parser.add_argument('--ckpt_path', type=str, default=None)


    # regularization
    parser.add_argument('--sd', type=float, default=0.1,
                        help='stochastic depth / droppath (https://paperswithcode.com/method/stochastic-depth)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')


    # data aug
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--resize_size', type=int, default=None, help='resize_size')
    parser.add_argument('--horizontal_flip', action='store_false',
                        help='use horizontal flip when training (on by default)')
    parser.add_argument('--rand_aug', action='store_true',
                        help='RandAugment augmentation used')
    parser.add_argument('--trivial_aug', action='store_true',
                        help='use trivialaugmentwide')
    parser.add_argument('--re', default=0.0, type=float,
                        help='Random Erasing probability (def: 0.25)')

    # cutmix and mixup (multi image data aug)
    parser.add_argument('--cm', action='store_true', help='Use Cutmix')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    parser.add_argument('--mu', action='store_true', help='Use Mixup')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')


    # results/wandb
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--run_name', type=str, default=None,
                        help='name for run in wandb')
    parser.add_argument('--project_name', type=str, default='RecognitionTutorial',
                        help='project folder in wandb')

    # inference
    parser.add_argument('--images_path', type=str,
                        default='data/ASDDataset/ASD/51160.jpg',
                        help='path to folder (with images) or image')
    parser.add_argument('--vis_mask', type=str, default=None,
                        help='vis mechanism: https://github.com/frgfm/torch-cam')

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.cfg:
        config = yaml_config_hook(os.path.abspath(args.cfg))
        for k, v in config.items():
            if hasattr(args, k):
                setattr(args, k, v)

    return args


def set_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return 0


def build_transform(args):
    # lots of different augmentations currently
    # https://pytorch.org/vision/stable/transforms.html
    args.resize_size = int(args.image_size // 0.875) if not args.resize_size else args.resize_size

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    t = []

    t.append(transforms.Resize(args.resize_size))
    t.append(transforms.RandomCrop([args.image_size, args.image_size]))

    if args.horizontal_flip:
        t.append(transforms.RandomHorizontalFlip())
    if args.rand_aug:
        t.append(transforms.RandAugment())
    if args.trivial_aug:
        t.append(transforms.TrivialAugmentWide())

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))

    if args.re:
        t.append(transforms.RandomErasing(p=args.re))

    train_transform = transforms.Compose(t)

    test_transform = transforms.Compose([
        transforms.Resize(args.resize_size),
        transforms.CenterCrop([args.image_size, args.image_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform


class DatasetImgTarget(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)


def build_dataloaders(args):
    train_transform, test_transform = build_transform(args)


    # choose dataset, download train and test splits if needed
    if args.dataset_name == 'cifar10':
        train_ds = datasets.CIFAR10(root=args.dataset_root_path, train=True,
                                    transform=train_transform, download=True)
        test_ds = datasets.CIFAR10(root=args.dataset_root_path, train=False,
                                    transform=test_transform, download=True)
        args.num_classes = 10
    elif args.dataset_name == 'cifar100':
        train_ds = datasets.CIFAR100(root=args.dataset_root_path, train=True,
                                    transform=train_transform, download=True)
        test_ds = datasets.CIFAR100(root=args.dataset_root_path, train=False,
                                    transform=test_transform, download=True)
        args.num_classes = 100
    else:
        train_ds = DatasetImgTarget(args, split='train', transform=train_transform)
        test_ds = DatasetImgTarget(args, split='test', transform=test_transform)
        args.num_classes = train_ds.num_classes


    setattr(args, f'num_images_train', train_ds.__len__())
    setattr(args, f'num_images_test', test_ds.__len__())
    print(f'''{args.dataset_name}
          N_train={args.num_images_train}
          N_test={args.num_images_test}
          K={args.num_classes}.''')

    # shuffle so that each epoch and each iteration are different
    # (stochasticness can be good for training deep learning models), drop_last for train speed
    # c, h, w -> b, c, h, w in an iterable way
    train_loader = data.DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_ds, args.batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


class TIMMModel(nn.Module):
    def __init__(self, model_name, pretrained=False, num_classes=10,
                 image_size=224, sd=0.0):
        super().__init__()

        assert (model_name in timm.list_models() or \
            model_name in timm.list_models(pretrained=True)), f'{model_name} not in timm'


        if 'vgg' in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
                global_pool='', pre_logits=False)
        elif any(model in model_name for model in ['vit', 'deit', 'trans']):
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
                img_size=image_size, drop_path_rate=sd, global_pool='')
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
                drop_path_rate=sd, global_pool='')


        _, d, bsd = self.get_out_features(image_size)


        if bsd:
            self.classifier = nn.Sequential(
                Reduce('b s d -> b d', 'mean'),
                nn.Linear(d, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                Reduce('b d h w -> b d', 'mean'),
                nn.Linear(d, num_classes)
            )

    @torch.no_grad()
    def get_out_features(self, image_size):
        x = torch.rand(2, 3, image_size, image_size)
        x = self.model(x)

        if len(x.shape) == 3:
            b, s, d = x.shape
            bsd = True
        elif len(x.shape) == 4:
            b, d, h, w = x.shape
            s = h * w
            bsd = False

        return s, d, bsd

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


def build_model(args):
    model = TIMMModel(args.model_name, args.pretrained, args.num_classes,
                        args.image_size, args.sd)

    model.to(args.device)
    model.zero_grad()
    print(model)

    if args.ckpt_path:
        state_dict = torch.load(
            args.ckpt_path, map_location=torch.device('cpu'))['model']

        ret = model.load_state_dict(state_dict, strict=False)

        print(f'Missing keys when loading pretrained weights: {ret.missing_keys}')
        print(f'Unexpected keys when loading pretrained weights: {ret.unexpected_keys}')

        print('Loaded from custom checkpoint.')

    return model


def train_loop(args, train_loader, model, criterion, optimizer):
    # certain modules behave differently during train/test (dropout)
    model.train()

    loss_epoch = 0
    correct = 0
    total = 0
    y_a = None

    for idx, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(args.device), labels.to(args.device)


        images, y_a, y_b, lam = get_mix(images, labels, train=True, args=args)
        # wandb.log({'images': wandb.Image(images)})


        # calculate outputs by running images through the network
        outputs = model(images)


        if y_a is not None:
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            loss = criterion(outputs, labels)


        # if loss is infinity then stop training
        assert math.isfinite(loss), f'Loss is not finite: {loss}, stopping training'


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_epoch += loss.item()


        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


        # optionally log at each iter or each x iters
        wandb.log({'train_loss': loss})
        if idx % args.log_freq == 0:
            acc_iter = 100 * (predicted == labels).sum().item() / images.shape[0]
            print(f'{idx} / {len(train_loader)}, Loss: {loss}, Acc@1: {acc_iter}')


    acc = round(100 * correct / total, 2)
    return loss_epoch, acc


def test_loop(args, test_loader, model):
    # certain modules behave differently during train/test (dropout)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(args.device), labels.to(args.device)

            # calculate outputs by running images through the network
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = round(100 * correct / total, 2)
    return acc


def set_results(args):
    # create directory for results
    if not args.run_name:
        args.run_name = f'{args.dataset_name}_{args.model_name}_{args.serial}'
    args.results_dir = os.path.join(args.results_dir, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)

    # logger
    wandb.init(config=args, project=args.project_name)
    wandb.run.name = args.run_name

    return 0


def train_end(args, model, test_acc, optimizer, time_start):
    # save checkpoint and corresponding args, acc and optimizer
    state = {
        'config': args,
        'model': model.state_dict(),
        'accuracy': test_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(args.results_dir, 'last.pth')) 

    # computational cost stats (time, params, memory)
    time_total = time.time() - time_start
    time_total = round(time_total / 60, 2)  # mins

    # count number of parameters in model
    no_params = sum([p.numel() for p in model.parameters()])
    no_params = round(no_params / (1e6), 2)  # millions of parameters

    # amount of memory used by model in GPU
    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    max_memory = round(max_memory, 2)

    wandb.run.summary['no_params'] = no_params
    wandb.run.summary['time_total'] = time_total
    wandb.run.summary['max_memory'] = max_memory
    wandb.run.summary['test_acc'] = test_acc
    wandb.finish()

    return 0


def setup_env(args):
    # data loaders for train and test
    train_loader, test_loader = build_dataloaders(args)

    # build our model
    model = build_model(args)

    # loss function and optimizer to guide training through backpropagation
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    return train_loader, test_loader, model, criterion, optimizer


def main():
    time_start = time.time()

    # options for training from command line
    args = parse_args()

    # set seed (to avoid randomness)
    set_random_seed(args.seed)

    train_loader, test_loader, model, criterion, optimizer = setup_env(args)

    # for saving results
    set_results(args)

    # train model: loop through data for x epochs
    for epoch in range(args.epochs):
        loss, train_acc = train_loop(args, train_loader, model, criterion, optimizer)

        # log to command line and wandb once per epoch
        print(f'Epoch {epoch} loss: {loss}, acc: {train_acc}')
        log_dic = {'epoch': epoch, 'train_loss': loss, 'train_acc': train_acc}
        wandb.log(log_dic)

    # evaluate on test data
    test_acc = test_loop(args, test_loader, model)
    print('Test accuracy: ', test_acc)

    # finish traininig routine
    train_end(args, model, test_acc, optimizer, time_start)

    return 0


if __name__ == '__main__':
    main()
