import torch

import math
import numpy as np


def get_mean(features, labels):
    class_mean = []
    lbl = torch.unique(labels)
    for i in lbl.tolist():
        class_mean.append(torch.mean(features[labels==i], dim=0).reshape(1, -1))
    class_mean = torch.vstack(class_mean)
    global_mean = torch.mean(class_mean, dim=0)

    return global_mean, class_mean


def equinorm(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)

    v_h = torch.norm(class_mean - global_mean, dim=1)
    res_h = torch.std(v_h) / torch.mean(v_h)

    v_w = torch.norm(weights, dim=1)
    res_w = torch.std(v_w) / torch.mean(v_w)

    return res_h, res_w


def equiangularity(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)

    v_nrm_h = class_mean - global_mean
    nrm_h = torch.matmul(v_nrm_h, v_nrm_h.T)

    v_denom_h = torch.norm(v_nrm_h, dim=1, keepdim=True)
    denom_h = torch.matmul(v_denom_h, v_denom_h.T)

    mask_h = torch.triu(torch.ones_like(nrm_h), diagonal=1)
    res_h = torch.std((nrm_h / denom_h)[mask_h != 0])

    nrm_w = torch.matmul(weights, weights.T)

    v_denom_w = torch.norm(weights, dim=1, keepdim=True)
    denom_w = torch.matmul(v_denom_w, v_denom_w.T)

    mask_w = torch.triu(torch.ones_like(nrm_w), diagonal=1)
    res_w = torch.std((nrm_w / denom_w)[mask_w != 0])

    return res_h, res_w


def maximal_angle(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)
    
    C = weights.shape[0]

    v_nrm_h = class_mean - global_mean
    nrm_h = torch.matmul(v_nrm_h, v_nrm_h.T)

    v_denom_h = torch.norm(v_nrm_h, dim=1, keepdim=True)
    denom_h = torch.matmul(v_denom_h, v_denom_h.T)

    mask_h = torch.triu(torch.ones_like(nrm_h), diagonal=1)
    res_h = torch.mean((nrm_h / denom_h)[mask_h != 0] + 1. / (C - 1))

    nrm_w = torch.matmul(weights, weights.T)

    v_denom_w = torch.norm(weights, dim=1, keepdim=True)
    denom_w = torch.matmul(v_denom_w, v_denom_w.T)

    mask_w = torch.triu(torch.ones_like(nrm_w), diagonal=1)
    res_w = torch.mean((nrm_w / denom_w)[mask_w != 0] + 1. / (C - 1))

    return res_h, res_w


def class_mean_convergence(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)

    M = class_mean - global_mean
    M_tilde = M / torch.norm(M)

    W = weights / torch.norm(weights)

    return torch.norm(W - M_tilde) ** 2


def within_class_variation(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)

    num_classes, num_features = weights.shape[0], weights.shape[1]

    Sigma_W = torch.zeros((num_features, num_features))
    for i in range(num_classes):
        v_c = (features[labels==i] - class_mean[i].reshape(1, -1)).unsqueeze(2)
        Sigma_W_c = torch.mean(torch.bmm(v_c, torch.transpose(v_c, 1, 2)), dim=0)
        Sigma_W += Sigma_W_c
    Sigma_W /= num_classes

    v_B = (class_mean - global_mean).unsqueeze(2)
    Sigma_B = torch.mean(torch.bmm(v_B, torch.transpose(v_B, 1, 2)), dim=0)

    return torch.trace(torch.matmul(Sigma_W, torch.linalg.pinv(Sigma_B)) / num_classes)


def ncc(features, labels, weights, global_mean=None, class_mean=None):
    if (global_mean is None) or (class_mean is None):
        global_mean, class_mean = get_mean(features, labels)

    pred = torch.argmin(torch.norm(features - class_mean.unsqueeze(1), dim=2).T, dim=1)

    return torch.mean((pred != labels).float())

