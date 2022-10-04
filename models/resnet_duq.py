import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.models import resnet18, resnet34

import pdb


class ResNet_DUQ(nn.Module):
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
        # print("z: ", z)
        z = torch.einsum("ij,mnj->imn", z, self.W)
        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        # print("diff: ", diff.shape)
        # pdb.set_trace()
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()
        # print("diff: ", diff)
        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        # print("inputs shape: ", x.shape)
        z = self.feature_extractor(x)
        # print('z.shape', z.shape)
        outputs = self.rbf(z)
        # print("outputs ", outputs[:10])
        # outputs = outputs / torch.sum(outputs, dim=1).unsqueeze(1)
        # print("outputs ", outputs[:10])
        # pdb.set_trace()
        predictions = torch.argmax(outputs, dim=1)
        # print("predictions: ", predictions)

        return outputs, predictions

def ResNet18_DUQ(centroid_size, model_output_size, num_classes=10):
    feature_extractor = resnet18()

    feature_extractor.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    model = ResNet_DUQ(
            feature_extractor,
            num_classes=num_classes,
            centroid_size=centroid_size,
            model_output_size=model_output_size,
            length_scale=0.1,
            gamma=0.999,
        )

    return model

def ResNet34_DUQ(centroid_size, model_output_size, num_classes):
    feature_extractor = resnet34()

    feature_extractor.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )

    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    model = ResNet_DUQ(
            feature_extractor,
            num_classes=num_classes,
            centroid_size=centroid_size,
            model_output_size=model_output_size,
            length_scale=0.1,
            gamma=0.999,
        )

    return model


if __name__ == '__main__':
    model_output_size = 512
    centroid_size = 512
    num_classes = 160

    model = ResNet34_DUQ(centroid_size, model_output_size, num_classes)

    inputs = torch.randn(256, 3, 64, 64)

    outputs, _ = model(inputs)
    print("outputs shape: ", outputs.shape)