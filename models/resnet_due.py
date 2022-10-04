import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import cluster

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from models.layers import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d

import pdb

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(
        self, 
        wrapped_conv,
        wrapped_batchnorm,
        input_size,
        in_planes,
        planes,
        stride=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = wrapped_batchnorm(planes)
        input_size = (input_size - 1) // stride + 1

        self.conv2 = wrapped_conv(input_size, planes, planes, kernel_size=3, stride=1)
        self.bn2 = wrapped_batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                wrapped_conv(input_size, in_planes, planes, kernel_size=1, stride=stride),
                wrapped_batchnorm(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_DUE(nn.Module):
    def __init__(
        self, 
        input_size,
        block, 
        num_blocks, 
        spectral_conv=True,
        spectral_bn=True,
        coeff=3,
        n_power_iterations=1,
    ):
        super(ResNet_DUE, self).__init__()

        def wrapped_bn(num_features):
            if spectral_bn:
                bn = SpectralBatchNorm2d(num_features, coeff)
            else:
                bn = nn.BatchNorm2d(num_features)
            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            if not spectral_conv:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1, input_size = self._make_layer(block, 64, num_blocks[0], 1, input_size)
        self.layer2, input_size = self._make_layer(block, 128, num_blocks[1], 2, input_size)
        self.layer3, input_size = self._make_layer(block, 256, num_blocks[2], 2, input_size)
        self.layer4, input_size = self._make_layer(block, 512, num_blocks[3], 2, input_size)

    def _make_layer(self, block, planes, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.wrapped_conv,
                    self.wrapped_bn,
                    input_size,
                    self.in_planes, 
                    planes, 
                    stride)
                )
            self.in_planes = planes * block.expansion
            input_size = (input_size - 1) // stride + 1

        return nn.Sequential(*layers), input_size

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out

def ResNet18_DUE_encoder(input_size):
    feature_extractor = ResNet_DUE(input_size=input_size, block=BasicBlock, num_blocks=[2, 2, 2, 2])
    return feature_extractor

def ResNet34_DUE_encoder(input_size):
    feature_extractor = ResNet_DUE(input_size=input_size, block=BasicBlock, num_blocks=[3, 4, 6, 3])
    return feature_extractor


