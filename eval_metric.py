import pdb

import numpy as np 

import torch
import torch.nn.functional as F

def one_hot(targets, num_classes):
    batch_size = targets.shape[0]
    targets_one_hot = torch.zeros(batch_size, num_classes).cuda()
    targets_one_hot[list(range(batch_size)), targets.squeeze()] = 1
    return targets_one_hot

def brier_score(outputs, targets, num_classes):
    probability = F.softmax(outputs, dim=1)
    targets_one_hot = one_hot(targets, num_classes)
    brier = torch.mean((probability - targets_one_hot) ** 2, dim=1)
    brier = torch.mean(brier)
    return brier

def nll_score(outputs, targets):
    criterion = torch.nn.NLLLoss()
    m = torch.nn.LogSoftmax(dim=1)
    nll = criterion(m(outputs), targets)
    return nll

def entropy_score(outputs):
    probability = F.softmax(outputs, dim=1)
    probability_log = torch.log(probability)
    entropy = - torch.sum(probability * probability_log, dim=1)
    entropy = torch.mean(entropy)
    return entropy

def brier_score_distance(outputs, targets, num_classes):
    probability = outputs / torch.sum(outputs, dim=1).unsqueeze(1)
    targets_one_hot = one_hot(targets, num_classes)
    brier = torch.mean((probability - targets_one_hot) ** 2, dim=1)
    brier = torch.mean(brier)
    return brier

def nll_score_distance(outputs, targets):
    criterion = torch.nn.NLLLoss()
    probability = outputs / torch.sum(outputs, dim=1).unsqueeze(1)
    probability_log = torch.log(probability)
    nll = criterion(probability_log, targets)
    return nll

def entropy_score_distance(outputs):
    outputs = outputs / torch.sum(outputs, dim=1).unsqueeze(1)
    outputs_log = torch.log(outputs)
    entropy = - torch.sum(outputs * outputs_log, dim=1)
    entropy = torch.mean(entropy)
    return entropy

def brier_score_due(probability, targets, num_classes):
    targets_one_hot = one_hot(targets, num_classes)
    brier = torch.mean((probability - targets_one_hot) ** 2, dim=1)
    brier = torch.mean(brier)
    return brier

def nll_score_due(probability, targets):
    criterion = torch.nn.NLLLoss()
    probability_log = torch.log(probability)
    nll = criterion(probability_log, targets)
    return nll

def entropy_score_due(probability):
    probability_log = torch.log(probability)
    entropy = - torch.sum(probability * probability_log, dim=1)
    entropy = torch.mean(entropy)
    return entropy

def reject_rate(outputs):
    probability = F.softmax(outputs, dim=1)
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    reject_rate_list = []
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        
        max_probabilty = torch.max(probability, dim=1)[0] > threshold
        reject_rate = 1 - torch.mean(max_probabilty.float())

        reject_rate_list.append(reject_rate.item())
        # pdb.set_trace()
    return reject_rate_list

def reject_rate_distance(outputs):
    probability = outputs / torch.sum(outputs, dim=1).unsqueeze(1)
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    reject_rate_list = []
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        
        max_probabilty = torch.max(probability, dim=1)[0] > threshold
        reject_rate = 1 - torch.mean(max_probabilty.float())

        reject_rate_list.append(reject_rate.item())

    return reject_rate_list

def reject_rate_due(probability):
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    reject_rate_list = []
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        
        max_probabilty = torch.max(probability, dim=1)[0] > threshold
        reject_rate = 1 - torch.mean(max_probabilty.float())

        reject_rate_list.append(reject_rate.item())

    return reject_rate_list

    