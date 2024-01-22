import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

__all__ = [
    'SimpleFNN',
    'LeNet5',
]

# reproduce FNN (http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)
class SimpleFNN(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(SimpleFNN, self).__init__()
        self.no_actv = cfg.backbone.no_actv
        self.in_features = cfg.backbone.in_features

        self.fc1 = nn.Linear(self.in_features, 500)
        self.fc2 = nn.Linear(500, 300)

    def forward(self, input):
        if input.dim() > 2:
            input = input.reshape(-1, self.in_features)

        x = self.fc1(input)
        x = F.relu(x, inplace=True)

        x = self.fc2(x)
        if not self.no_actv:
            x = F.relu(x, inplace=True)

        return x


# reproduce LeNet-5 (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
class LeNet5(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LeNet5, self).__init__()
        self.no_actv = cfg.backbone.no_actv
        self.in_channels = cfg.backbone.in_channels

        self.conv1 = nn.Conv2d(self.in_channels, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5), padding=0)
        self.fc3 = nn.Linear(16*5*5, 120)
        self.fc4 = nn.Linear(120, 84)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = torch.flatten(x, 1)

        x = self.fc3(x)
        x = F.relu(x, inplace=True)

        x = self.fc4(x)
        if not self.no_actv:
            x = F.relu(x, inplace=True)

        return x
        
        
