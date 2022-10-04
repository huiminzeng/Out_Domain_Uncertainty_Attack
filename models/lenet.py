import torch
from torch import nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
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

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        # print("after conv 1 shape: ", x.shape)
        x = self.conv2(x)
        # print("after conv 2 shape: ", x.shape)
        x = self.fc_1(x.view(batch_size,-1))
        outputs = self.fc_2(x)

        probability = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probability, dim=-1)

        return outputs, predictions

if __name__ == '__main__':

    model = LeNet()

    inputs = torch.randn(4, 1, 28, 28)

    outputs = model(inputs)
    print("outputs shape: ", outputs.shape)