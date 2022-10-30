import torch
import torch.nn as nn
import os
from utils import GeM
from torchvision import models


# We intentionally make the model shallow as there is more info about
# image texture in first layers than in last ones, which are more abstract.
class Regressor(nn.Module):
    def __init__(self, dim=32):
        super(Regressor, self).__init__()

        self.dim = dim

        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, padding=1, padding_mode="reflect")
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(
            dim, dim * 2, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn2 = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(
            dim * 2, dim * 4, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn3 = nn.BatchNorm2d(dim * 4)
        self.conv4 = nn.Conv2d(
            dim * 4, dim * 8, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn4 = nn.BatchNorm2d(dim * 8)
        self.conv5 = nn.Conv2d(
            dim * 8, dim * 16, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn5 = nn.BatchNorm2d(dim * 16)
        self.conv6 = nn.Conv2d(
            dim * 16, dim * 32, kernel_size=3, padding=1, padding_mode="reflect"
        )
        self.bn6 = nn.BatchNorm2d(dim * 32)
        self.fc1 = nn.Linear(dim * 32, 1)

        self.pool = nn.AvgPool2d(kernel_size=4, stride=2)
        self.global_pool = GeM()
        self.act = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.act(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.act(self.bn5(self.conv5(x)))
        x = self.pool(x)
        x = self.act(self.bn6(self.conv6(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))
        return x


class Pretrained_regressor(nn.Module):
    def __init__(self):
        super(Pretrained_regressor, self).__init__()

        os.environ["TORCH_HOME"] = "/tmp"
        os.environ["ENV_TORCH_HOME"] = "/tmp"
        net = models.resnet18(progress=False, pretrained=True)

        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.global_pool = GeM()
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        # for param in self.conv1.parameters():
        #     param.requires_grad = False
        # for param in self.bn1.parameters():
        #     param.requires_grad = False
        # for param in self.layer1.parameters():
        #     param.requires_grad = False
        # for param in self.layer2.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc(x))
        return x
