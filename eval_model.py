from __future__ import print_function
import argparse
import os
import shutil
import time
import pdb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from dataloader import get_dataloader, get_datasets
from models import *
from eval_metric import *

from adversary import adversary
from adversary import adversary_pgd
from adversary import adversary_duq
from adversary import adversary_due

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

parser = argparse.ArgumentParser(description='PyTorch Training')

# dataset and dataloader
parser.add_argument('--data_path', default='../../Data', type=str)
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--out_domain', default=True, type=bool)
parser.add_argument('-b', '--batch-size', default=200, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')

parser.add_argument('--cuda', '-c', default=True)

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--model', default='LeNet', type=str, metavar='Model',
                    help='model type: LeNet, ResNet-18, LeNet_DUQ')

# test specification
parser.add_argument('--load_path', default='trained_models', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

parser.add_argument('--mode', default='baseline', type=str)
parser.add_argument('--seed', default=0, type=int)

# number of ensembles
parser.add_argument('--num_ensembles', default=5, type=int)

# ensemble adversarial training specification
parser.add_argument('--epsilon', default=0.1, type=float)
parser.add_argument('--epsilon_attack', default=0.1, type=float)

# DUQ training specification
parser.add_argument('--lambda_reg', default=0.1, type=float)
parser.add_argument('--centroid_size', default=0, type=int)
parser.add_argument('--model_output_size', default=0, type=int)

# DUE training specification
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--n_inducing_points', default=10, type=int)
parser.add_argument("--kernel", default="RBF", type=str)

# attack evaluation
parser.add_argument('--targeted', default=False, type=bool)
parser.add_argument('--target', default=0, type=int)

def main():
    global args
    args = parser.parse_args()

    # create Light CNN for face recognition
    if args.model == 'LeNet' or args.model == 'LeNet_DUQ' or args.model == 'LeNet_DUE' or args.model == 'LeNet_SNGP':
        args.dataset = 'mnist'

    elif args.model == 'ResNet-18' or args.model == 'ResNet-18_DUQ' or args.model == 'ResNet-18_DUE' or args.model == 'ResNet-18_SNGP': 
        args.dataset = 'cifar'
        
    else:
        print('Error model type\n')

    # pdb.set_trace()
    cudnn.benchmark = True

    #load image
    train_loader, val_loader, test_loader_out = get_dataloader(args)
    # pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    if args.mode == 'baseline':
        load_dir = os.path.join(args.load_path, 'baseline', args.model, 'seed_1')
        load_name_model = os.path.join(load_dir, 'model_best.checkpoint')
        print("We are evaluating baseline model!")
        validate_baseline(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'deep_ensemble':
        load_root = os.path.join(args.load_path, 'deep_ensemble', args.model)
        load_name_model = []
        for i in range(args.num_ensembles):
            load_dir = os.path.join(load_root, 'seed_'+str(i))
            load_name_model_each = os.path.join(load_dir, 'model_best.checkpoint')
            load_name_model.append(load_name_model_each)
        print("We are evaluating deep ensemble model!")
        validate_ensemble(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'deep_ensemble_at':
        load_root = os.path.join(args.load_path, 'deep_ensemble_at', args.model, 'eps_' + str(args.epsilon))
        load_name_model = []
        for i in range(args.num_ensembles):
            load_dir = os.path.join(load_root, 'seed_'+str(i))
            load_name_model_each = os.path.join(load_dir, 'model_best.checkpoint')
            load_name_model.append(load_name_model_each)
        print("We are evaluating deep ensemble adversarial training model!")
        validate_ensemble(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'attack_deep_ensemble':
        load_root = os.path.join(args.load_path, 'deep_ensemble', args.model)
        load_name_model = []
        for i in range(args.num_ensembles):
            load_dir = os.path.join(load_root, 'seed_'+str(i))
            load_name_model_each = os.path.join(load_dir, 'model_best.checkpoint')
            load_name_model.append(load_name_model_each)
        print("We are attacking deep ensemble model!")

        load_name_model_wb = os.path.join(args.load_path, 'baseline', args.model, 'seed_10', 'model_best.checkpoint')
        validate_attack_ensemble(load_name_model_wb, load_name_model, test_loader_out, args)

    elif args.mode == 'attack_deep_ensemble_at':
        load_root = os.path.join(args.load_path, 'deep_ensemble_at', args.model, 'eps_' + str(args.epsilon))
        load_name_model = []
        for i in range(args.num_ensembles):
            load_dir = os.path.join(load_root, 'seed_'+str(i))
            load_name_model_each = os.path.join(load_dir, 'model_best.checkpoint')
            load_name_model.append(load_name_model_each)
        print("We are attacking deep ensemble adversarial training model!")

        load_name_model_wb = os.path.join(args.load_path, 'baseline', args.model, 'seed_1', 'model_best.checkpoint')
        validate_attack_ensemble(load_name_model_wb, load_name_model, test_loader_out, args)

    elif args.mode == 'duq':
        load_name_model = os.path.join(args.load_path, 'DUQ_backup', args.model, 'lambda_reg_' + str(args.lambda_reg), 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are evaluating duq model!")

        validate_duq(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'attack_duq':
        load_name_model = os.path.join(args.load_path, 'DUQ', args.model, 'lambda_reg_' + str(args.lambda_reg), 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are attacking duq model!")

        validate_attack_duq(load_name_model, test_loader_out, args)

    elif args.mode == 'due':
        load_name_model = os.path.join(args.load_path, 'DUE', args.model, 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are evaluating due model!")

        validate_due(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'attack_due':
        load_name_model = os.path.join(args.load_path, 'DUE', args.model, 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are attacking due model!")

        validate_attack_due(load_name_model, test_loader_out, args)

    elif args.mode == 'sngp':
        load_name_model = os.path.join(args.load_path, 'SNGP', args.model, 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are evaluating sngp model!")

        validate_sngp(load_name_model, val_loader, test_loader_out, args)

    elif args.mode == 'attack_sngp':
        load_name_model = os.path.join(args.load_path, 'SNGP', args.model, 'seed_' + str(args.seed), 'model_best.checkpoint')
        print("We are attacking sngp model!")

        validate_attack_sngp(load_name_model, test_loader_out, args)


    print("TESTING IS OVER!!!!!")
    

def validate_baseline(load_name_model, val_loader, test_loader_out, args):
    # load model
    if args.model == 'LeNet':
        model = LeNet().cuda()
    elif args.model == 'ResNet-18':
        model = ResNet18().cuda()
    elif args.model == 'ResNet-34':
        model = ResNet34().cuda()

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    # switch to evaluate mode
    model.eval()

    # in-domain test
    brier_his = []
    nll_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)
        
        brier = brier_score(outputs, targets, args.num_classes)
        nll = nll_score(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        brier_his.append(brier.item())
        nll_his.append(nll.item())
        acc_his.append(acc.item())

    print('In-Domain Test set: Brier: {:.5f}, Nll: {:.3f}, Accuracy: {:.3f}'.format(np.mean(brier_his), np.mean(nll_his), np.mean(acc_his)))

    # out-domain test
    entropy_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)
        probability = F.softmax(outputs, dim=1)
        
        entropy = entropy_score(outputs)
        acc = torch.mean((predictions==targets).float())

        entropy_his.append(entropy.item())
        acc_his.append(acc.item())

    print('Out-Domain Test set: Entropy: {:.3f}, Accuracy: {:.3f}'.format(np.mean(entropy_his), np.mean(acc_his)))
    
def validate_ensemble(load_name_model, val_loader, test_loader_out, args):
    # load model
    save_root = os.path.join('ensemble_tensors', args.mode, args.model, 'eps_train_' + str(args.epsilon))
    # pdb.set_trace()

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    if len(os.listdir(save_root)) >= args.num_ensembles:
        evaluted = True
    else:
        evaluted = False

    evaluted = False
    if not evaluted:
        if args.model == 'LeNet':
            model = LeNet().cuda()
        elif args.model == 'ResNet-18':
            model = ResNet18().cuda()
        elif args.model == 'ResNet-34':
            model = ResNet34().cuda()
            args.num_classes = 160

        for i in range(args.num_ensembles):
            save_dir = os.path.join('ensemble_tensors', args.mode, args.model, 'eps_train_' + str(args.epsilon), 'seed_' + str(i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("we are saving ensemble tensors to: ", save_dir)

            load_name_model_each = load_name_model[i]
            if os.path.isfile(load_name_model_each):
                print("=> loading model checkpoint '{}'".format(load_name_model_each))
                checkpoint_model = torch.load(load_name_model_each)
                model.load_state_dict(checkpoint_model['state_dict'])    
                print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
                
            else:
                print("=> no checkpoint found at '{}'".format(load_name_model))

            # switch to evaluate mode
            model.eval()
            # in-domain test
            for j, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.cuda()

                # compute output
                outputs, predictions = model(inputs)
                outputs = outputs.detach().cpu()

                torch.save(outputs, os.path.join(save_dir, 'outputs_tensor_' + str(j) + '.pt'))
                torch.save(targets, os.path.join(save_dir, 'targets_tensor_' + str(j) + '.pt'))

            # pdb.set_trace()
            # out-domain test
            for j, (inputs, targets) in enumerate(test_loader_out):
                inputs = inputs.cuda()
            
                # compute output
                outputs, predictions = model(inputs)
                outputs = outputs.detach().cpu()

                torch.save(outputs, os.path.join(save_dir, 'outputs_out_tensor_' + str(j) + '.pt'))
                torch.save(targets, os.path.join(save_dir, 'targets_out_tensor_' + str(j) + '.pt'))
            # pdb.set_trace()

    # in-domain test
    num_batches = len(val_loader)
    brier_his = []
    nll_his = []
    acc_his = []
    for i in range(num_batches):
        outputs_ensemble = 0
        for j in range(args.num_ensembles):
            save_dir = os.path.join('ensemble_tensors', args.mode, args.model, 'eps_train_' + str(args.epsilon), 'seed_' + str(j))

            outputs = torch.load(os.path.join(save_dir, 'outputs_tensor_' + str(i) + '.pt')).cuda()
            targets = torch.load(os.path.join(save_dir, 'targets_tensor_' + str(i) + '.pt')).cuda()

            outputs_ensemble += outputs
        
        # pdb.set_trace()
        outputs_ensemble = outputs_ensemble / args.num_ensembles
        brier = brier_score(outputs_ensemble, targets, args.num_classes)
        nll = nll_score(outputs_ensemble, targets)
        acc = torch.mean((torch.argmax(outputs_ensemble, dim=1) == targets).float())
        
        brier_his.append(brier.item())
        nll_his.append(nll.item())
        acc_his.append(acc.item())

    print('In-Domain Test set: Brier: {:.5f}, Nll: {:.3f}, Accuracy: {:.3f}'.format(np.mean(brier_his), np.mean(nll_his), np.mean(acc_his)))

    # in-domain test
    num_batches_out = len(test_loader_out)
    entropy_his = []
    reject_score_his = []
    for i in range(num_batches_out):
        outputs_out_ensemble = 0
        for j in range(args.num_ensembles):
            save_dir = os.path.join('ensemble_tensors', args.mode, args.model, 'eps_train_' + str(args.epsilon), 'seed_' + str(j))

            outputs = torch.load(os.path.join(save_dir, 'outputs_out_tensor_' + str(i) + '.pt')).cuda()
            targets = torch.load(os.path.join(save_dir, 'targets_out_tensor_' + str(i) + '.pt')).cuda()

            outputs_out_ensemble += outputs

        outputs_out_ensemble = outputs_out_ensemble / args.num_ensembles
        # probability = F.softmax(outputs_out_ensemble, dim=1)

        entropy = entropy_score(outputs_out_ensemble)
        reject_score_list = reject_rate(outputs_out_ensemble)

        # print("entropy ensemble: ", entropy.item())
        entropy_his.append(entropy.item())
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])
       
def validate_duq(load_name_model, val_loader, test_loader_out, args):
    # load model
    if args.model == 'LeNet_DUQ':
        torch.manual_seed(args.seed)
        args.model_output_size = 84
        args.centroid_size = args.model_output_size // 2
        args.dataset = 'mnist'
        args.num_classes = 10

        model = LeNet_DUQ(centroid_size = args.centroid_size, 
                          model_output_size = args.model_output_size).cuda()

    elif args.model == 'ResNet-18_DUQ':
        torch.manual_seed(args.seed)
        args.model_output_size = 512
        args.centroid_size = args.model_output_size // 2
        args.dataset = 'cifar'
        args.num_classes = 10

        model = ResNet18_DUQ(centroid_size = args.centroid_size, 
                             model_output_size = args.model_output_size).cuda()
        
    else:
        print('Error model type\n')

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    save_dir = os.path.join('ensemble_tensors', args.mode, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving ensemble tensors attacked to: ", save_dir)

    # switch to evaluate mode
    model.eval()

    # in-domain test
    brier_his = []
    nll_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)

        brier = brier_score_distance(outputs, targets, 10)
        nll = nll_score_distance(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        brier_his.append(brier.item())
        nll_his.append(nll.item())
        acc_his.append(acc.item())

    print('In-Domain Test set: Brier: {:.5f}, Nll: {:.3f}, Accuracy: {:.3f}'.format(np.mean(brier_his), np.mean(nll_his), np.mean(acc_his)))

    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, predictions = model(inputs)
    
        # entropy = entropy_score(outputs)

        torch.save(outputs / torch.sum(outputs, dim=1).unsqueeze(1), os.path.join(save_dir, 'outputs_out_tensor_' + str(i) + '.pt'))
            
        entropy = entropy_score_distance(outputs)
        acc = torch.mean((predictions==targets).float())

        entropy_his.append(entropy.item())
        acc_his.append(acc.item())

        reject_score_list = reject_rate_distance(outputs)

        # print("entropy ensemble: ", entropy.item())
        entropy_his.append(entropy.item())
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])

def validate_due(load_name_model, val_loader, test_loader_out, args):
    # load model
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

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    # pdb.set_trace()
    likelihood = SoftmaxLikelihood(num_classes=args.num_classes, mixing_weights=False)
    likelihood = likelihood.cuda()

    elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_set))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    save_dir = os.path.join('ensemble_tensors', args.mode, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving ensemble tensors attacked to: ", save_dir)

    # switch to evaluate mode
    model.eval()
    likelihood.eval()
    # in-domain test
    brier_his = []
    nll_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)

        # Sample softmax values independently for classification at test time
        outputs = outputs.to_data_independent_dist()
        # The mean here is over likelihood samples
        probability = likelihood(outputs).probs.mean(0)
        predictions = torch.argmax(probability, dim=-1)

        # measure accuracy and record loss
        acc = torch.mean((predictions==targets).float())

        brier = brier_score_due(probability, targets, args.num_classes)
        nll = nll_score_due(probability, targets)
        acc = torch.mean((predictions==targets).float())

        brier_his.append(brier.item())
        nll_his.append(nll.item())
        acc_his.append(acc.item())

    print('In-Domain Test set: Brier: {:.5f}, Nll: {:.3f}, Accuracy: {:.3f}'.format(np.mean(brier_his), np.mean(nll_his), np.mean(acc_his)))

    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs = model(inputs)

        outputs = outputs.to_data_independent_dist()
        probability = likelihood(outputs).probs.mean(0)

        torch.save(probability, os.path.join(save_dir, 'outputs_out_tensor_' + str(i) + '.pt'))
            
        # entropy = entropy_score(outputs)
        entropy = entropy_score_due(probability)

        predictions = torch.argmax(probability, dim=-1)
        acc = torch.mean((predictions==targets).float())

        entropy_his.append(entropy.item())
        acc_his.append(acc.item())

        reject_score_list = reject_rate_due(probability)

        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])

def validate_sngp(load_name_model, val_loader, test_loader_out, args):
    # pdb.set_trace()
    if args.model == 'LeNet_SNGP':
        torch.manual_seed(args.seed)
        args.dataset = 'mnist'
        args.num_classes = 10
        input_size = 28
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = LeNet_DUE_encoder(input_size).cuda()

        num_deep_features = 84
        num_gp_features = 256
        normalize_gp_features = True
        num_random_features = 256
        num_data = len(train_set)
        mean_field_factor = 25
        ridge_penalty = 1
        lengthscale = 2

        model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            args.num_classes,
            num_data,
            args.batch_size,
            mean_field_factor,
            ridge_penalty,
            lengthscale,
        ).cuda()

    elif args.model == 'ResNet-18_SNGP':
        torch.manual_seed(args.seed)
        args.dataset = 'cifar'
        args.num_classes = 10
        input_size = 32
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = ResNet18_DUE_encoder(input_size).cuda()

        num_deep_features = 512
        num_gp_features = 512
        normalize_gp_features = True
        num_random_features = 512
        num_data = len(train_set)
        mean_field_factor = 25
        ridge_penalty = 1
        lengthscale = 2

        model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            args.num_classes,
            num_data,
            args.batch_size,
            mean_field_factor,
            ridge_penalty,
            lengthscale,
        ).cuda()
        
    else:
        print('Error model type\n')

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    save_dir = os.path.join('ensemble_tensors', args.mode, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving ensemble tensors attacked to: ", save_dir)
    
    # switch to evaluate mode
    model.eval()
    # in-domain test
    brier_his = []
    nll_his = []
    acc_his = []
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, _ = model(inputs)
        probability = F.softmax(outputs, dim=1)

        torch.save(probability, os.path.join(save_dir, 'outputs_out_tensor_' + str(i) + '.pt'))
         
        predictions = torch.argmax(probability, dim=-1)

        brier = brier_score(outputs, targets, args.num_classes)
        nll = nll_score(outputs, targets)
        acc = torch.mean((predictions==targets).float())

        brier_his.append(brier.item())
        nll_his.append(nll.item())
        acc_his.append(acc.item())

    print('In-Domain Test set: Brier: {:.5f}, Nll: {:.3f}, Accuracy: {:.3f}'.format(np.mean(brier_his), np.mean(nll_his), np.mean(acc_his)))

    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # compute output
        outputs, _ = model(inputs)

        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=-1)
        
        entropy = entropy_score(outputs)
        entropy_his.append(entropy.item())

        acc = torch.mean((predictions==targets).float())
        acc_his.append(acc.item())

        reject_score_list = reject_rate(outputs)
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain ADV Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])

def validate_attack_ensemble(load_name_model_wb, load_name_model, test_loader_out, args):
    # load model
    if args.model == 'LeNet':
        model = LeNet().cuda()
        model_wb = LeNet().cuda()
    elif args.model == 'ResNet-18':
        model = ResNet18().cuda()
        model_wb = ResNet18().cuda()
    elif args.model == 'ResNet-34':
        model = ResNet34().cuda()
        # model_wb = ResNet18().cuda()

    # print("=> loading white-box model checkpoint '{}'".format(load_name_model_wb))
    # checkpoint_model_wb = torch.load(load_name_model_wb)
    # model_wb.load_state_dict(checkpoint_model_wb['state_dict'])    
    # print("=> loaded white-box checkpoint !! in-domain acc: {}".format(checkpoint_model_wb['prec1']))

    if args.targeted:
        attack_type = 'targeted_' + str(args.target)
    else:
        attack_type = 'untargeted'

    for i in range(args.num_ensembles):
        save_dir = os.path.join('ensemble_tensors', args.mode, attack_type, args.model, 'eps_train_' + str(args.epsilon), 'eps_attack_' + str(args.epsilon_attack), 'seed_' + str(i))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("we are saving ensemble tensors attacked to: ", save_dir)
        load_name_model_each = load_name_model[i]
        if os.path.isfile(load_name_model_each):
            print("=> loading model checkpoint '{}'".format(load_name_model_each))
            checkpoint_model = torch.load(load_name_model_each)
            model.load_state_dict(checkpoint_model['state_dict'])    
            print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
            
        else:
            print("=> no checkpoint found at '{}'".format(load_name_model))

        # switch to evaluate mode
        model.eval()

        for j, (inputs, targets) in enumerate(test_loader_out):
            
            # visualization_save_dir = os.path.join('visualization', args.mode, args.model, 'eps_' + str(args.epsilon))
            # if not os.path.exists(visualization_save_dir):
            #     os.makedirs(visualization_save_dir)

            # torch.save(inputs, os.path.join(visualization_save_dir, 'inputs_out_tensor_' + str(j) + '.pt'))
            
            inputs = inputs.cuda()
            # inputs_adv = adversary(model_wb, inputs, args)

            inputs_adv = adversary(model, inputs, args)
            # torch.save(inputs_adv, os.path.join(visualization_save_dir, 'inputs_out_adv_tensor_' + str(j) + '.pt'))
            
            outputs_adv, _ = model(inputs_adv)
            outputs_adv = outputs_adv.detach().cpu()
            probability_adv = F.softmax(outputs_adv, dim=1)

            probability_adv_log = torch.log(probability_adv)
            entropy_adv = - torch.sum(probability_adv * probability_adv_log, dim=1)
            entropy_adv = torch.mean(entropy_adv)
            # print("entropy out adv: ", entropy_adv.item())
            # pdb.set_trace()
            
            torch.save(outputs_adv, os.path.join(save_dir, 'outputs_out_adv_tensor_' + str(j) + '.pt'))
            # break
    # in-domain test
    num_batches_out = len(test_loader_out)
    entropy_his = []
    reject_score_his = []
    for i in range(num_batches_out):
        outputs_out_ensemble = 0
        for j in range(args.num_ensembles):
            save_dir = os.path.join('ensemble_tensors', args.mode, attack_type, args.model, 'eps_train_' + str(args.epsilon), 'eps_attack_' + str(args.epsilon_attack), 'seed_' + str(j))
            outputs = torch.load(os.path.join(save_dir, 'outputs_out_adv_tensor_' + str(i) + '.pt')).cuda()
            
            # probability = F.softmax(outputs, dim=1)
            # probability_log = torch.log(probability)
            # entropy = - torch.sum(probability * probability_log, dim=1)
            # entropy = torch.mean(entropy)
            # print("entropy: ", entropy.item())

            outputs_out_ensemble += outputs
        
        outputs_out_ensemble = outputs_out_ensemble / args.num_ensembles

        entropy = entropy_score(outputs_out_ensemble)
        reject_score_list = reject_rate(outputs_out_ensemble)

        # print("entropy ensemble: ", entropy.item())
        entropy_his.append(entropy.item())
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain ADV Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])

def validate_attack_duq(load_name_model, test_loader_out, args):
    # load model
    if args.model == 'LeNet_DUQ':
        torch.manual_seed(args.seed)
        args.model_output_size = 84
        args.centroid_size = args.model_output_size // 2
        args.dataset = 'mnist'
        args.num_classes = 10

        model = LeNet_DUQ(centroid_size = args.centroid_size, 
                          model_output_size = args.model_output_size).cuda()

    elif args.model == 'ResNet-18_DUQ':
        torch.manual_seed(args.seed)
        args.model_output_size = 512
        args.centroid_size = args.model_output_size // 2
        args.dataset = 'cifar'
        args.num_classes = 10

        model = ResNet18_DUQ(centroid_size = args.centroid_size, 
                             model_output_size = args.model_output_size).cuda()
        
    else:
        print('Error model type\n')

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    if args.targeted:
        attack_type = 'targeted_' + str(args.target)
    else:
        attack_type = 'untargeted'

    save_dir = os.path.join('ensemble_tensors', args.mode, attack_type, args.model, 'eps_' + str(args.epsilon))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # switch to evaluate mode
    model.eval()

    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()
        # print("inputs shape: ", inputs.shape)
        # compute output
        inputs_adv = adversary_duq(model, inputs, targets, args)

        outputs_adv, _ = model(inputs_adv)
        
        torch.save(outputs_adv, os.path.join(save_dir, 'outputs_out_adv_tensor_' + str(i) + '.pt'))
            
        # entropy = entropy_score(outputs_adv)
        entropy = entropy_score_distance(outputs_adv)

        reject_score_list = reject_rate_distance(outputs_adv)

        # print("entropy ensemble: ", entropy.item())
        entropy_his.append(entropy.item())
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])


def validate_attack_due(load_name_model, test_loader_out, args):
    # load model
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

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    likelihood = SoftmaxLikelihood(num_classes=args.num_classes, mixing_weights=False)
    likelihood = likelihood.cuda()

    elbo_fn = VariationalELBO(likelihood, gp, num_data=len(train_set))
    loss_fn = lambda x, y: -elbo_fn(x, y)

    if args.targeted:
        attack_type = 'targeted_' + str(args.target)
    else:
        attack_type = 'untargeted'

    save_dir = os.path.join('ensemble_tensors', args.mode, attack_type, args.model, 'eps_' + str(args.epsilon))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # switch to evaluate mode
    model.eval()
    likelihood.eval()
    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()
        # print("inputs shape: ", inputs.shape)
        # compute output

        inputs_adv = adversary_due(model, inputs, targets, likelihood, loss_fn, args)

        outputs_adv = model(inputs_adv)

        # Sample softmax values independently for classification at test time
        outputs_adv = outputs_adv.to_data_independent_dist()
        # The mean here is over likelihood samples
        probability_adv = likelihood(outputs_adv).probs.mean(0)

        torch.save(probability_adv, os.path.join(save_dir, 'outputs_out_adv_tensor_' + str(i) + '.pt'))
            
        # entropy = entropy_score(outputs_adv)
        entropy = entropy_score_distance(probability_adv)
        entropy_his.append(entropy.item())

        reject_score_list = reject_rate_due(probability_adv)
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))

    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])

def validate_attack_sngp(load_name_model, test_loader_out, args):
    if args.model == 'LeNet_SNGP':
        torch.manual_seed(args.seed)
        args.dataset = 'mnist'
        args.num_classes = 10
        input_size = 28
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = LeNet_DUE_encoder(input_size).cuda()

        num_deep_features = 84
        num_gp_features = 256
        normalize_gp_features = True
        num_random_features = 256
        num_data = len(train_set)
        mean_field_factor = 25
        ridge_penalty = 1
        lengthscale = 2

        model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            args.num_classes,
            num_data,
            args.batch_size,
            mean_field_factor,
            ridge_penalty,
            lengthscale,
        ).cuda()

    elif args.model == 'ResNet-18_SNGP':
        torch.manual_seed(args.seed)
        args.dataset = 'cifar'
        args.num_classes = 10
        input_size = 32
        #load image
        train_set, val_set = get_datasets(args)

        feature_extractor = ResNet18_DUE_encoder(input_size).cuda()

        num_deep_features = 512
        num_gp_features = 512
        normalize_gp_features = True
        num_random_features = 512
        num_data = len(train_set)
        mean_field_factor = 25
        ridge_penalty = 1
        lengthscale = 2

        model = Laplace(
            feature_extractor,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            args.num_classes,
            num_data,
            args.batch_size,
            mean_field_factor,
            ridge_penalty,
            lengthscale,
        ).cuda()

    else:
        print('Error model type\n')

    if os.path.isfile(load_name_model):
        print("=> loading model checkpoint '{}'".format(load_name_model))
        checkpoint_model = torch.load(load_name_model)
        model.load_state_dict(checkpoint_model['state_dict'])    
        print("=> loaded checkpoint !! in-domain acc: {}".format(checkpoint_model['prec1']))
        
    else:
        print("=> no checkpoint found at '{}'".format(load_name_model))

    
    if args.targeted:
        attack_type = 'targeted_' + str(args.target)
    else:
        attack_type = 'untargeted'

    save_dir = os.path.join('ensemble_tensors', args.mode, attack_type, args.model, 'eps_' + str(args.epsilon))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # switch to evaluate mode
    model.eval()
    # out-domain test
    entropy_his = []
    acc_his = []
    reject_score_his = []
    for i, (inputs, targets) in enumerate(test_loader_out):
        inputs = inputs.cuda()
        targets = targets.cuda()
        # print("inputs shape: ", inputs.shape)
        # compute output

        inputs_adv = adversary_pgd(model, inputs, args)

        outputs_adv, _ = model(inputs_adv)

        torch.save(outputs_adv, os.path.join(save_dir, 'outputs_out_adv_tensor_' + str(i) + '.pt'))

        entropy = entropy_score(outputs_adv)
        entropy_his.append(entropy.item())

        reject_score_list = reject_rate(outputs_adv)
        reject_score_his.append(reject_score_list)
        # pdb.set_trace()

    reject_score_his = np.array(reject_score_his)
    reject_score_his = np.mean(reject_score_his, axis=0)
    print('Out-Domain ADV Test set: Entropy: {:.3f}'.format(np.mean(entropy_his)))
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print('Reject Score List: ', reject_score_his[3:])


if __name__ == '__main__':
    main()