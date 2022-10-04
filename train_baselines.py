from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import pdb

from models import *

from dataloader import get_dataloader

parser = argparse.ArgumentParser(description='PyTorch Training')

# dataset and dataloader
parser.add_argument('--data_path', default='../Data', type=str)
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--out_domain', default=False, type=bool)

parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--model', default='ResNet-18', type=str, metavar='Model',
                    help='model type: LeNet, ResNet-18')

# training specification
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--save_path', default='trained_models', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

parser.add_argument('--mode', default='baseline', type=str)
parser.add_argument('--seed', default=10, type=int)

def main():
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_path, args.mode, args.model, 'seed_' + str(args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving adv model to: ", save_dir)
    
    if args.model == 'LeNet':
        torch.manual_seed(args.seed)
        model = LeNet().cuda()
        args.dataset = 'mnist'

    elif args.model == 'ResNet-18':
        torch.manual_seed(args.seed)
        model = ResNet18().cuda()
        args.dataset = 'cifar'
        
    else:
        print('Error model type\n')

    params= list(model.parameters())
    
    optimizer= torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True

    #load image
    train_loader, val_loader = get_dataloader(args)
    # pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    best_prec = 0
    best_model = None
    best_epoch = 0
    best_optimizer= None
    
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion)

        # save checkpoint
        if acc > best_prec:
            best_prec = acc
            best_model = model.state_dict()
            best_epoch = epoch
            best_optimizer = optimizer.state_dict()

            save_name_model = os.path.join(save_dir, 'model_best.checkpoint')
            model_best_checkpoint = {'epoch': best_prec + 1,
                                    'optimizer': best_optimizer,
                                    'state_dict': best_model,
                                    'prec1': best_prec,
                                        }
            torch.save(model_best_checkpoint, save_name_model)

        last_save_name_model = os.path.join(save_dir, 'model_last.checkpoint')
        model_last_checkpoint = {'epoch': epoch + 1,
                                        'optimizer': optimizer.state_dict(),
                                        'state_dict': model.state_dict(),
                                        'prec1': acc,
                                            }
        torch.save(model_last_checkpoint, last_save_name_model)

    print("TRAINING IS OVER!!!!!")
    
def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)
        
        # measure accuracy and record loss
        loss = criterion(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pdb.set_trace()
        
        if i % args.print_freq == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                epoch, i, len(train_loader), loss.item(), acc.item()*100))

        # break


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    loss_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)
        
        # measure accuracy and record loss
        loss = criterion(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        loss_his.append(loss.item())
        acc_his.append(acc.item())

    print('\nTest set: Average loss: {}, Accuracy: ({})\n'.format(np.mean(loss_his), np.mean(acc_his)))

    return np.mean(acc_his)
    

def adjust_learning_rate(optimizer, epoch):
    scale = 0.1
    step  = 30
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

if __name__ == '__main__':
    main()