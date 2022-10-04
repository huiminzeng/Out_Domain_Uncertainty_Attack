import os
import pdb
import glob
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pdb
import json 

import numpy as np

class NotMNIST_dataset(Dataset):
    def __init__(self, data_path):
        # specifying the zip file name 
        num_imgs = 1000
        dir_root = os.path.join(data_path, "notMNIST_small")
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        self.data_path = []
        self.data_label = []
        for i in range(10):
            dir_name = os.path.join(dir_root, labels[i])
            data_path_temp = sorted(glob.glob(dir_name + '/*.png'))[:num_imgs]
            self.data_path += data_path_temp

            data_label_temp = [i] * num_imgs
            self.data_label += data_label_temp

        self.transform = transforms.Compose(
                [transforms.ToTensor()])
                    
    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_tensor = self.transform(image_set)
        image_label = self.data_label[idx]

        return image_tensor, image_label

def get_dataloader(args):
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    if args.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                        transform=transforms.ToTensor()),
                                        batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(args.data_path, train=False,
                                        transform=transforms.ToTensor()),
                                        batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.out_domain:
            print("Using NotMNIST!")
            test_loader_out = torch.utils.data.DataLoader(NotMNIST_dataset(args.data_path),
                                        batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader, test_loader, test_loader_out

        else:
            return train_loader, test_loader

    elif args.dataset == 'cifar':

        transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ])

        transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(args.data_path, train=True, download=True, 
                                        transform=transform_train), 
                                        batch_size=args.batch_size, shuffle=True, **kwargs)

     
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, 
                                        transform=transform_test), 
                                        batch_size=args.batch_size, shuffle=False, **kwargs)

        if args.out_domain:
            print("Using SVHN!")
            test_loader_out = torch.utils.data.DataLoader(
                torchvision.datasets.SVHN(os.path.join(args.data_path, 'SVHN'), split='test', download=True, 
                                        transform=transform_test),
                                        batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader, test_loader, test_loader_out

        else:
            return train_loader, test_loader

def get_datasets(args):
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    if args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(args.data_path, train=True, download=True,
                                        transform=transforms.ToTensor())

        test_set = torchvision.datasets.MNIST(args.data_path, train=False,
                                        transform=transforms.ToTensor())
                                        
        return train_set, test_set


    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ])

        transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                    ])

        train_set = torchvision.datasets.CIFAR10(args.data_path, train=True, download=True, 
                                        transform=transform_train)

     
        test_set = torchvision.datasets.CIFAR10(args.data_path, train=False, download=True, 
                                        transform=transform_test)

        return train_set, test_set

if __name__ == '__main__':
    pass