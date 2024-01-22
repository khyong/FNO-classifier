import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

__all__ = [
    'CrossEntropyCustom',
    'BCECustom',
    'MSECustom',
    'CosineLoss',
    'LogLoss',
    'LogPosLoss',
    'LDAMLoss',
    'LabelAwareSmoothing',
]


class CrossEntropyCustom(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super(CrossEntropyCustom, self).__init__()

    def forward(self, output, targets):
        return F.cross_entropy(output, targets)


class BCECustom(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super().__init__()
        self.num_classes = param_dict['num_classes']

    def forward(self, output, targets):
        tgt = F.one_hot(targets, num_classes=self.num_classes).float()
        return F.binary_cross_entropy(output, tgt)


class MSECustom(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super().__init__()

    def forward(self, features, weights):
        return F.mse_loss(features, weights)


class CosineLoss(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super().__init__()
        self.num_classes = param_dict['num_classes']

    def forward(self, output, targets):
        tgt = F.one_hot(targets, num_classes=self.num_classes).float()
        return torch.mean(torch.sum(tgt * (1. - output), dim=1))


class LogLoss(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super(LogLoss, self).__init__()
        self.num_classes = param_dict['num_classes']
        self.eps = 1e-18

    def forward(self, output, targets):
        tgt = F.one_hot(targets, num_classes=self.num_classes).float()
        x = torch.log(output + self.eps)
        return -torch.mean(torch.sum(tgt * x, dim=1))


class LogPosLoss(nn.Module):
    def __init__(self, param_dict=None, **kwargs):
        super(LogPosLoss, self).__init__()
        self.num_classes = param_dict['num_classes']
        self.eps = 1e-18

    def forward(self, output, targets):
        tgt = F.one_hot(targets, num_classes=self.num_classes).float()
        output = (output + 1) / 2.
        x = torch.log(output + self.eps)
        return -torch.mean(torch.sum(tgt * x, dim=1))


# The LDAMLoss class is copied from the official PyTorch implementation in LDAM
# (https://github.com/kaidic/LDAM-DRW)
class LDAMLoss(nn.Module):
    def __init__(self, param_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = param_dict['num_class_list']
        self.rank = param_dict['rank']

        cfg = param_dict['cfg']
        max_m = cfg.loss.LDAM.max_margin
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).cuda(self.rank)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.step_epoch = cfg.loss.LDAM.drw_epoch
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch - 1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).cuda(self.rank)

    def forward(self, x, label, **kwargs):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, label.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.cuda(self.rank)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, label, weight=self.weight)


# The LabelAwareSmoothing class is copied from the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
class LabelAwareSmoothing(nn.Module):
    def __init__(self, param_dict=None):
        super(LabelAwareSmoothing, self).__init__()
        self.num_class_list = param_dict['num_class_list']
        self.rank = param_dict['rank']

        cfg = param_dict['cfg']
        smooth_head = cfg.loss.LAS.smooth_head
        smooth_tail = cfg.loss.LAS.smooth_tail
        shape = cfg.loss.LAS.shape
        power = cfg.loss.LAS.power

        n_1 = max(self.num_class_list)
        n_k = min(self.num_class_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(self.num_class_list) - n_k) * np.pi / (2 * (n_1 - n_k)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(self.num_class_list) - n_k) / (n_1 - n_k)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(self.num_class_list) - n_k) * np.pi / (2 * (n_1 - n_k)))
        elif (shape == 'exp') and (power is not None):
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(self.num_class_list) - n_k) / (n_1 - n_k), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        self.smooth = self.smooth.cuda(self.rank)

    def forward(self, x, label, **kwargs):
        label = label.cuda(self.rank)
        smoothing = self.smooth[label]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=label.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()

