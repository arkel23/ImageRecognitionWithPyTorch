import os
import time
import random
import argparse

import torch
from torch import nn
from torch.nn import functional as F

import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms


def parse_args():
    parser = argparse.ArgumentParser('Arguments for code')
    # training in general
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='freq for logging train results')
    # data: dataset and image size
    parser.add_argument('--dataset_name', default='cifar10', type=str,
                        help='dataset name')
    parser.add_argument('--dataset_root_path', type=str, default='data',
                        help='the root directory for where the data files are')
    parser.add_argument('--image_size', type=int, default=32, help='image_size')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    # model
    parser.add_argument('--model_name', type=str, default='lenet')
    # results
    parser.add_argument('--results_dir', type=str, default='results')

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return 0


def build_dataloaders(args):
    # 32x32 -> 36x36 -random crop-> 32x32
    # lots of different augmentations currently
    resize_size = int(args.image_size // 0.875)
    train_transform = transforms.Compose([
        # from PIL import Image
        # img = Image.open('our_image.jpg')
        # resizing and cropping can be done in different ways: 
        # square/short side resize -> random crop is common in fgir
        transforms.Resize([resize_size, resize_size]),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        # commonly resize then center crop (assume RoI is in center and
        # background on borders) 
        # but in cifar usually directly resize
        transforms.Resize([args.image_size, args.image_size]),
        # transforms.CenterCrop(args.image_size)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

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

    # shuffle so that each epoch and each iteration are different
    # (stochasticness can be good for training deep learning models), drop_last for train speed
    # c, h, w -> b, c, h, w in an iterable way
    train_loader = data.DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_ds, args.batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


class LeNet(nn.Module):
    def __init__(self, num_classes=10, image_size=32):
        super().__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        # out_size = ((in_size + 2*padding - (kernel_size - 1) - 1) // stride) + 1
        # out_size = ((32 + 2*0 - (5 - 1) - 1) // 1) + 1 = 28
        # round division down
        # 32 -> 28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)

        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # kernel_size, stride, padding=0
        # out_size = ((in_size + 2*padding - (kernel_size - 1) - 1 // stride) + 1)
        # 28 -> (26 // 2) + 1 -> 14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 14 -> 10 -pooled-> 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)

        # traditional nets required knowing input size to predict fc size
        spatial_size = ((((image_size - 4) // 2) - 4) // 2)
        self.fc1 = nn.Linear(16 * spatial_size * spatial_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model(args):
    if args.model_name == 'lenet':
        model = LeNet(args.num_classes, args.image_size)

    model.to(args.device)
    model.zero_grad()
    print(model)

    return model


def train_loop(args, train_loader, model, criterion, optimizer):
    # certain modules behave differently during train/test (dropout)
    model.train()

    loss_epoch = 0
    correct = 0
    total = 0

    for idx, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(args.device), labels.to(args.device)

        # calculate outputs by running images through the network
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_epoch += loss.item()

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # optionally log at each iter or each x iters
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


def main():
    # options for training from command line
    args = parse_args()

    # set seed (to avoid randomness)
    set_random_seed(args.seed)

    # data loaders for train and test
    train_loader, test_loader = build_dataloaders(args)

    # build our model
    model = build_model(args)

    # loss function and optimizer to guide training through backpropagation
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # create directory for results
    os.makedirs(os.path.join(args.results_dir, args.run_name), exist_ok=True)

    # train model: loop through data for x epochs
    for epoch in range(args.epochs):
        loss, train_acc = train_loop(args, train_loader, model, criterion, optimizer)

        # log to command line once per epoch
        print(f'Epoch {epoch} loss: {loss}, acc: {train_acc}')

    # evaluate on test data
    test_acc = test_loop(args, test_loader, model)
    print('Test accuracy: ', test_acc)

    # save checkpoint and corresponding args, acc and optimizer
    state = {
        'config': args,
        'model': model.state_dict(),
        'accuracy': test_acc,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(args.results_dir, args.run_name, 'last.pth')) 

    return 0


if __name__ == '__main__':
    main()
