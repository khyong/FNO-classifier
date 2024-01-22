import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

__all__ = [
    'OrthLinear',
    'OrthDenseLinear',
    'OrthSparseLinear',
    'OrthHalfLinear',
    'OrthWLinear',
    'OrthDupLinear',
]


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
            if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
                if num_features-1 in feat_indices[i]:
                    feat_indices[i] = np.delete(
                        feat_indices[i], np.where(feat_indices[i] == num_features-1))
        if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
            base[:,-1] = 0.
        base = F.normalize(base, dim=1)

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices


class OrthDenseLinear(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        orth_linear = torch.nn.utils.parametrizations.orthogonal(
            nn.Linear(num_features, num_classes))

        base = orth_linear.weight.detach().cpu().clone()
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.base, None)


class OrthSparseLinear(nn.Module):
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
        if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
            base[:,-1] = 0.
        w = torch.rand(1, num_features)

        self.feat_indices = feat_indices
        self.register_buffer('base', w * base)

    def forward(self, x):
        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices


class OrthHalfLinear(nn.Module):
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

        mask = torch.zeros(num_classes, num_features).long() 
        base = torch.zeros(num_classes, num_features)
        for i in range(num_classes):
            base[i][feat_indices[i]] = 1
            v = torch.rand(len(feat_indices[i]))
            v = torch.where(v < 0.5, 1, -1).long()
            mask[i][feat_indices[i]] = v
        if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
            mask[:,-1] = 0.
            base[:,-1] = 0.
        base = mask * F.normalize(base, dim=1)

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices


class OrthWLinear(nn.Module):
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
        if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
            base[:,-1] = 0.
        base = F.normalize(base, dim=1)
        #self.w = nn.Parameter(torch.ones(1, num_features))
        self.w = nn.Parameter(torch.ones(1))

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        w = F.relu(self.w)
        w = w * self.base
        #x = F.linear(x, w, None)
        #return F.normalize(x, dim=1)
        return F.linear(x, w, None)

    def get_feat_indices(self):
        return self.feat_indices


class OrthDupLinear(nn.Module):
    def __init__(self, num_features, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.sparse_factor = kwargs['cfg'].classifier.sparse_factor
        #self.sparse_factor = kwargs['sparse_factor']

        rnd_indices = np.random.permutation(np.arange(num_features))
        n = num_features // num_classes if self.sparse_factor == -1 else self.sparse_factor
        feat_indices = [rnd_indices[i*n:(i+1)*n] for i in range(num_classes)]
        rem_indices = rnd_indices[num_classes*n:]

        base = torch.zeros(num_classes, num_features)
        for i in range(num_classes):
            base[i][feat_indices[i]] = 1
        base[:,rem_indices] = 0.001
        if ('cfg' in kwargs) and ('Sph' in kwargs['cfg'].reshape.type):
            base[:,-1] = 0.
        base = F.normalize(base, dim=1)

        self.feat_indices = feat_indices
        self.register_buffer('base', base)

    def forward(self, x):
        return F.linear(x, self.base, None)

    def get_feat_indices(self):
        return self.feat_indices

