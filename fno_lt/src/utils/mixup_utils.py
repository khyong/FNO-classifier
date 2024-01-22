import torch

import numpy as np
from collections import Counter


def get_index_randomly(batch_size, y, rank=None):
    if rank is not None:
        index = torch.randperm(batch_size).cuda(rank)
    else:
        index = torch.randperm(batch_size)
    
    return index
        

def get_index_disjointly(batch_size, y, rank=None):
    idx = np.zeros(batch_size, dtype=np.int64)
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, list):
        y_np = np.array(y)
    else:
        y_np = y
    classes_in_batch = torch.unique(y).detach().cpu().numpy()

    for cls_num in classes_in_batch:
        tgt_idx = np.where(y_np==cls_num)[0]
        other_idx = np.where(y_np!=cls_num)[0]
        if len(tgt_idx) > len(other_idx):
            num_dups = len(tgt_idx) // len(other_idx) + 1
            dup_idx = np.concatenate([other_idx] * num_dups)
            idx[tgt_idx] = np.random.permutation(dup_idx)[:len(tgt_idx)]
        else:
            idx[tgt_idx] = np.random.permutation(other_idx)[:len(tgt_idx)]

    if rank is not None:
        index = torch.from_numpy(idx).cuda(rank)
    else:
        index = torch.from_numpy(idx)

    return index


def get_indices_balancely(batch_size, y, rank=None):
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    elif isinstance(y, list):
        y_np = np.array(y)
    else:
        y_np = y
    classes_in_batch = torch.unique(y).detach().cpu().numpy()

    counter = Counter(y_np)
    prob = {k: 1. / (len(counter) * v) for k, v in counter.items()}
    prob = [prob[v] for v in y_np]
    indices_a = np.random.choice(batch_size, batch_size, p=prob)
    y_a_np = y_np[indices_a]

    indices_b = np.zeros(batch_size, dtype=np.int64)
    for cls_num in classes_in_batch:
        counter = Counter(y_np[y_np!=cls_num])
        prob = {k: 1. / (len(counter) * v) for k, v in counter.items()}
        prob = [prob[v] for v in y_np[y_np!=cls_num]]
        tgt_idx = np.where(y_a_np==cls_num)[0]
        other_idx = np.where(y_np!=cls_num)[0]
        indices_b[tgt_idx] = np.random.choice(other_idx, size=len(tgt_idx), p=prob)

    if rank is not None:
        indices_a = torch.from_numpy(indices_a).cuda(rank)
        indices_b = torch.from_numpy(indices_b).cuda(rank)
    else:
        indices_a = torch.from_numpy(indices_a)
        indices_b = torch.from_numpy(indices_b)

    return indices_a, indices_b


# The mixup_data, mixup_criterion method is copied from 
# the official PyTorch implementation in Mixup
# => https://github.com/facebookresearch/mixup-cifar10.git
def mixup_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = get_index_randomly(batch_size, y, rank=rank)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# arc_mixup
def arc_mixup_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = get_index_randomly(batch_size, y, rank=rank)

    pi = 3.141592
    arc_lam = np.sin(pi/2 * lam)

    mixed_x = arc_lam * x + np.sqrt(1 - arc_lam**2) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, arc_lam

def arc_mixup_criterion(criterion, pred, y_a, y_b, arc_lam):
    return arc_lam * criterion(pred, y_a) + np.sqrt(1 - arc_lam**2) * criterion(pred, y_b)


# arc_mixup_cbs_sync
def arc_mixup_cbs_sync_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    indices_a, indices_b = get_indices_balancely(batch_size, y, rank=rank)

    pi = 3.141592
    arc_lam = np.sin(pi/2 * lam)

    mixed_x = arc_lam * x[indices_a] + np.sqrt(1 - arc_lam**2) * x[indices_b]
    y_a, y_b = y[indices_a], y[indices_b]
    return mixed_x, y_a, y_b, arc_lam

arc_mixup_cbs_sync_criterion = arc_mixup_criterion


# arc_mixup_cls
def arc_mixup_cls_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    index = get_index_disjointly(batch_size, y, rank=None)

    pi = 3.141592
    arc_lam = np.sin(pi/2 * lam)
    norm_lam = arc_lam / (arc_lam + np.sqrt(1 - arc_lam**2))

    mixed_x = norm_lam * x + (1 - norm_lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, arc_lam

def arc_mixup_cls_criterion(cls_mat, features, y_a, y_b, arc_lam):
    cls_a, cls_b = cls_mat[y_a], cls_mat[y_b]
    mixed_cls = arc_lam * cls_a + np.sqrt(1 - arc_lam**2) * cls_b
    prob = torch.sum(features * mixed_cls, dim=-1, keepdim=True)
    return -torch.mean(torch.log(prob + 1e-18))


# arc_mixup_cls_cbs
def arc_mixup_cls_cbs_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    indices_a, indices_b = get_indices_balancely(batch_size, y, rank=rank)

    pi = 3.141592
    arc_lam = np.sin(pi/2 * lam)
    norm_lam = arc_lam / (arc_lam + np.sqrt(1 - arc_lam**2))

    mixed_x = norm_lam * x[indices_a] + (1 - norm_lam) * x[indices_b]
    y_a, y_b = y[indices_a], y[indices_b]
    return mixed_x, y_a, y_b, arc_lam

arc_mixup_cls_cbs_criterion = arc_mixup_cls_criterion


# arc_mixup_cls_cbs_dup
arc_mixup_cls_cbs_dup_data = arc_mixup_cls_cbs_data

def arc_mixup_cls_cbs_dup_criterion(cls_mat, features, y_a, y_b, arc_lam):
    # if cls_mat is not an orthogonal matrix, use this.
    cls_a, cls_b = cls_mat[y_a], cls_mat[y_b]
    mixed_cls = torch.sqrt(arc_lam**2 * cls_a**2 + (1-arc_lam**2) * cls_b**2)
    prob = torch.sum(features * mixed_cls, dim=-1, keepdim=True)
    return -torch.mean(torch.log(prob + 1e-18))


# arc_mixup_cls_cbs_sync
def arc_mixup_cls_cbs_sync_data(x, y, alpha=1.0, rank=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    batch_size = x.size()[0]
    indices_a, indices_b = get_indices_balancely(batch_size, y, rank=rank)

    pi = 3.141592
    arc_lam = np.sin(pi/2 * lam)

    mixed_x = arc_lam * x[indices_a] + np.sqrt(1 - arc_lam**2) * x[indices_b]
    y_a, y_b = y[indices_a], y[indices_b]
    return mixed_x, y_a, y_b, arc_lam

arc_mixup_cls_cbs_sync_criterion = arc_mixup_cls_criterion

