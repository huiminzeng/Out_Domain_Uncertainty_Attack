import argparse
import os
import shutil
import time
import copy 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import pdb

from models import *

from dataloader import get_dataloader

def adversary(model, x_natural, args):

    device = x_natural.device
    criterion_ce = nn.CrossEntropyLoss().to(device)
        
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    torch.manual_seed(0)
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = x_natural.detach().clone()
    x_adv.requires_grad_()
    with torch.enable_grad():
        outputs_adv, _ = model(x_adv)
        
        if args.targeted:
            # print("targeted attack !!!")
            y = torch.ones(batch_size).long().cuda() * args.target
            loss = criterion_ce(outputs_adv, y)
        else:
            # print("untargeted attack !!!")
            y = torch.argmax(outputs_adv, dim=1)
            loss = criterion_ce(outputs_adv, y)
        
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() - args.epsilon_attack * torch.sign(grad.detach())
        # x_adv = x_adv.detach() - 0.063 * torch.sign(grad.detach())

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    return x_adv

def adversary_pgd(model, x_natural, args):

    device = x_natural.device
    criterion_ce = nn.CrossEntropyLoss().to(device)
        
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    torch.manual_seed(0)
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    
    for _ in range(10):
        # model_copy = copy.deepcopy(model)
        perturbation = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            outputs_adv, _ = model(x_adv)
            
            if args.targeted:
                # print("targeted attack !!!")
                y = torch.ones(batch_size).long().cuda() * args.target
                loss = criterion_ce(outputs_adv, y)
            else:
                # print("untargeted attack !!!")
                y = torch.argmax(outputs_adv, dim=1)
                loss = criterion_ce(outputs_adv, y)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() - args.epsilon_attack / 10 * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon_attack), x_natural + args.epsilon_attack)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # pdb.set_trace()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    return x_adv
    
def adversary_duq(model, x_natural, y, args):
    device = x_natural.device
    criterion_ce = nn.CrossEntropyLoss().to(device)
        
    model.eval()
    batch_size = len(x_natural)
    
    # generate adversarial example
    torch.manual_seed(0)
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    # x_adv = x_natural.detach().clone()
    # x_adv.requires_grad_()
    # with torch.enable_grad():
    #     outputs_adv, _ = model(x_adv)
        
    #     if args.targeted:
    #         # print("targeted attack !!!")
    #         y_onehot = torch.ones(batch_size, args.num_classes).cuda()
    #         y_onehot[:, args,target] = 1
    #         loss = F.binary_cross_entropy(outputs_adv, y_onehot, reduction="mean")

    #     else:
    #         # print("untargeted attack !!!")
    #         y = torch.argmax(outputs_adv, dim=1)
    #         y_onehot = F.one_hot(y, args.num_classes).float()
    #         loss = F.binary_cross_entropy(outputs_adv, y_onehot, reduction="mean")
        
    #     grad = torch.autograd.grad(loss, [x_adv])[0]
    #     x_adv = x_adv.detach() - args.epsilon * torch.sign(grad.detach())
    #     # x_adv = x_adv.detach() - 0.01 * torch.sign(grad.detach())

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    
    for _ in range(10):
        perturbation = x_adv - x_natural
        x_adv.requires_grad_()
        with torch.enable_grad():
            outputs_adv, _ = model(x_adv)
            
            if args.targeted:
                # print("targeted attack !!!")
                y_onehot = torch.ones(batch_size, args.num_classes).cuda()
                y_onehot[:, args,target] = 1
                loss = F.binary_cross_entropy(outputs_adv, y_onehot, reduction="mean")

            else:
                # print("untargeted attack !!!")
                y = torch.argmax(outputs_adv, dim=1)
                y_onehot = F.one_hot(y, args.num_classes).float()
                loss = F.binary_cross_entropy(outputs_adv, y_onehot, reduction="mean")
            
            # if args.targeted:
            #     # print("targeted attack !!!")
            #     y = torch.ones(batch_size).long().cuda() * args.target
            #     loss = criterion_ce(outputs_adv, y)
            # else:
            #     # print("untargeted attack !!!")
            #     y = torch.argmax(outputs_adv, dim=1)
            #     loss = criterion_ce(outputs_adv, y)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() - args.epsilon_attack / 10 * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon_attack), x_natural + args.epsilon_attack)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # outputs, _ = model(x_natural)
    # probability = F.softmax(outputs, dim=1)
    # probability_log = torch.log(probability)
    # entropy = - torch.sum(probability * probability_log, dim=1)
    
    # outputs_adv, _ = model(x_adv)
    # probability_adv = F.softmax(outputs_adv, dim=1)
    # probability_adv_log = torch.log(probability_adv)
    # entropy_adv = - torch.sum(probability_adv * probability_adv_log, dim=1)
    
    # pdb.set_trace()

    return x_adv

def adversary_due(model, x_natural, y, likelihood, loss_fn, args):
    device = x_natural.device

    model.eval()
    likelihood.eval()

    batch_size = len(x_natural)
    
    # outputs_natural = model(x_natural)
    # outputs_natural = outputs_natural.to_data_independent_dist()
    # probability = likelihood(outputs_natural).probs.mean(0)
    # predictions = torch.argmax(probability, dim=-1)

    # generate adversarial example
    torch.manual_seed(0)
    # x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = x_natural.detach().clone()
    x_adv.requires_grad_()

    # with torch.enable_grad():
    #     outputs_adv = model(x_adv)
        
    #     if args.targeted:
    #         # print("targeted attack !!!")
    #         y = torch.ones(batch_size).long().cuda() * args.target
    #         loss = loss_fn(outputs_adv, targets)

    #     else:
    #         # print("untargeted attack !!!")
    #         # loss = loss_fn(outputs_adv, predictions_clean)
    #         loss = loss_fn(outputs_adv, y)

    #     grad = torch.autograd.grad(loss, [x_adv])[0]
    #     x_adv = x_adv.detach() - args.epsilon * torch.sign(grad.detach())
    #     # x_adv = x_adv.detach() - 0.01 * torch.sign(grad.detach())

    for _ in range(10):
        x_adv.requires_grad_()
        with torch.enable_grad():
            outputs_adv = model(x_adv)
            outputs_adv = outputs_adv.to_data_independent_dist()
            probability = likelihood(outputs_adv).probs.mean(0)
            predictions = torch.argmax(probability, dim=-1)

            if args.targeted:
                # print("targeted attack !!!")
                y = torch.ones(batch_size).long().cuda() * args.target
                loss = loss_fn(outputs_adv, y)

            else:
                # print("untargeted attack !!!")
                # loss = loss_fn(outputs_adv, predictions_clean)
                loss = loss_fn(outputs_adv, predictions)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() - args.epsilon_attack / 10 * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - args.epsilon_attack), x_natural + args.epsilon_attack)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            # print("loss: ", loss.item())

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    # outputs = model(x_natural)
    # outputs = outputs.to_data_independent_dist()
    # probability = likelihood(outputs).probs.mean(0)
    # probability_log = torch.log(probability)
    # entropy = - torch.sum(probability * probability_log, dim=1)
    
    # outputs_adv = model(x_adv)
    # outputs_adv = outputs_adv.to_data_independent_dist()
    # probability_adv = likelihood(outputs_adv).probs.mean(0)
    # probability_adv_log = torch.log(probability_adv)
    # entropy_adv = - torch.sum(probability_adv * probability_adv_log, dim=1)
    
    # pdb.set_trace()

    return x_adv
