from __future__ import print_function
import argparse
import os
import shutil
import time
import pdb

import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from models import *

from dataloader import get_dataloader, get_datasets

parser = argparse.ArgumentParser(description='PyTorch Training')

# dataset and dataloader
parser.add_argument('--data_path', default='../Data', type=str)
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--out_domain', default=False, type=bool)

parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--model', default='ResNet-18_DUE', type=str, metavar='Model',
                    help='model type: LeNet_DUE, ResNet-18_DUE')

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

parser.add_argument('--mode', default='DUE', type=str)
parser.add_argument('--seed', default=0, type=int)

# DUE training specification
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--n_inducing_points', default=10, type=int)
parser.add_argument("--kernel", default="RBF", type=str)

def main():
    global args
    args = parser.parse_args()

    if args.model == 'LeNet_DUE':
        torch.manual_seed(args.seed)
        args.dataset = 'mnist'
        args.num_classes = 10
        input_size = 28
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = LeNet_DUE_encoder(input_size).cuda()

        initial_inducing_points, initial_lengthscale = initial_values(
                train_set, feature_extractor, args.n_inducing_points
            )

        gp = GP(
            num_outputs=args.num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args.kernel,
        )
    
        model = DKL(feature_extractor, gp).cuda()

    elif args.model == 'ResNet-18_DUE':
        torch.manual_seed(args.seed)
        args.dataset = 'cifar'
        args.num_classes = 10
        input_size = 32
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = ResNet18_DUE_encoder(input_size).cuda()

        initial_inducing_points, initial_lengthscale = initial_values(
                train_set, feature_extractor, args.n_inducing_points
            )

        gp = GP(
            num_outputs=args.num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            kernel=args.kernel,
        )
    
        model = DKL(feature_extractor, gp).cuda()

    else:
        print('Error model type\n')

    save_dir = os.path.join(args.save_path, args.mode, args.model, 'seed_' + str(args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving adv model to: ", save_dir)

    params= list(model.parameters())
    
    optimizer= torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True
    
    #load image
    train_loader, val_loader = get_dataloader(args)
    
    # pdb.set_trace()
    likelihood = SoftmaxLikelihood(num_classes=args.num_classes, mixing_weights=False)
    likelihood = likelihood.cuda()

    elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_set))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    best_prec = 0
    best_model = None
    best_epoch = 0
    best_optimizer= None
    
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, likelihood, loss_fn, optimizer, epoch, args)

        # evaluate on validation set
        acc = validate(val_loader, model, likelihood, loss_fn)

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

        # break

    print("TRAINING IS OVER!!!!!")

def train(train_loader, model, likelihood, loss_fn, optimizer, epoch, args):

    model.train()
    likelihood.train()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        if inputs.shape[0] != args.batch_size:
            continue
        targets = targets.cuda()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            # Sample softmax values independently for classification at test time
            outputs = outputs.to_data_independent_dist()
            # The mean here is over likelihood samples
            probability = likelihood(outputs).probs.mean(0)
            predictions = torch.argmax(probability, dim=-1)
            acc = torch.mean((predictions==targets).float())
            # pdb.set_trace()
            print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                epoch, i, len(train_loader), loss.item(), acc.item()*100))

        # break

def validate(val_loader, model, likelihood, loss_fn):
    # switch to evaluate mode
    model.eval()
    likelihood.eval()

    loss_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Sample softmax values independently for classification at test time
        outputs = outputs.to_data_independent_dist()

        # The mean here is over likelihood samples
        probability = likelihood(outputs).probs.mean(0)

        predictions = torch.argmax(probability, dim=-1)
        # measure accuracy and record loss
        
        acc = torch.mean((predictions==targets).float())

        loss_his.append(loss.item())
        acc_his.append(acc.item())

        # break

    print('\nTest set: Average loss: {:.3f}, Accuracy: ({:.3f})\n'.format(np.mean(loss_his), np.mean(acc_his)))

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
