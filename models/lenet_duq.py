import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.models import resnet18, resnet34

import pdb


class LeNet_extractor(nn.Module):
    def __init__(self):
        super(LeNet_extractor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=84),
            nn.ReLU(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc_1(x.view(batch_size,-1))

        return out


class lenet_duq(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum
        
        # pdb.set_trace()

    def forward(self, x):
        z = self.feature_extractor(x)
        outputs = self.rbf(z)

        predictions = torch.argmax(outputs, dim=1)

        return outputs, predictions

def LeNet_DUQ(centroid_size, model_output_size):
    feature_extractor = LeNet_extractor()

    model = lenet_duq(
            feature_extractor,
            num_classes=10,
            centroid_size=centroid_size,
            model_output_size=model_output_size,
            length_scale=0.5,
            gamma=0.999,
        )

    return model

if __name__ == '__main__':
    model_output_size = 84
    centroid_size = 32

    model = LeNet_DUQ(centroid_size, model_output_size)

    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)
    print("outputs shape: ", outputs.shape)