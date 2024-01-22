import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OrthLinear(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        rnd_indices = np.random.permutation(np.arange(num_features))
        n = num_features // num_classes
        feat_indices = [rnd_indices[i*n:(i+1)*n] for i in range(num_classes)]
        rem = num_features % num_classes
        rem_indices = np.random.permutation(np.arange(num_classes))[:rem]
        for i, rem_idx in enumerate(rem_indices):
            feat_indices[rem_idx] = np.append(
                feat_indices[rem_idx], rnd_indices[num_classes*n+i])

        base = torch.zeros(num_classes, num_features)
        for i in range(num_classes):
            base[i][feat_indices[i]] = 1
        base = F.normalize(base, dim=1)
        self.w = nn.Parameter(torch.ones(1))  # for softmax

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.w * self.base, None)  # for softmax
#        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices
