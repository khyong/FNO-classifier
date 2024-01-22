import torch
import torch.nn as nn

__all__ = [
    'LearnableWeightScaling',
]


# The LearnableWeightScaling class is copied from the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        out = self.learned_norm * x
        return out

