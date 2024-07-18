import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

__all__ = [
    'FNOClassifier',
    'ETFClassifier',
]


class FNOClassifier(nn.Module):
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

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices


class ETFClassifier(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        P = self.generate_random_orthogonal_matrix(num_features, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(float(num_classes) / (num_classes - 1)) * torch.matmul(P, I-((1./num_classes) * one))
        self.ori_M = M

        self.BN_H = nn.BatchNorm1d(num_features)

    def generate_random_orthogonal_matrix(self, num_features, num_classes):
        a = np.random.random(size=(num_features, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        return x

