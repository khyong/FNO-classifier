import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GlobalAvgPooling',
]


class GlobalAvgPooling(nn.Module):
    """Global Average Pooling
        Widely used in ResNet, Inception, DenseNet, etc.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        return x

