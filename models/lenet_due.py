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

class LeNet_DUE_encoder(nn.Module):
    def __init__(
        self, 
        input_size,
        coeff=3,
        n_power_iterations=1
    ):
        super(LeNet_DUE_encoder, self).__init__()

        def wrapped_conv(input_size, in_channels, out_channels, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_channels, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.conv1 = nn.Sequential(
            wrapped_conv(input_size, in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            wrapped_conv(12, in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        fc_temp = nn.Linear(in_features=16*4*4, out_features=84)
        self.fc_1 = nn.Sequential(
            spectral_norm_fc(fc_temp, coeff, n_power_iterations),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc_1(x.view(batch_size,-1))

        return out
