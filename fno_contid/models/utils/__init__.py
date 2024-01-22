import torch
import numpy as np

num_classes = {
    'seq-mnist': 2,
    'seq-cifar10': 2,
    'seq-cifar100': 10,
    'seq-tinyimg': 20,
}

mask_size = {
    'seq-mnist': 10,
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
}


def masking(outputs, labels, device, num_classes=2, mask_size=10, zero=False):
    for i in range(len(outputs)):
        t = ((labels[i] // num_classes) * num_classes).to('cpu')
        mask = [i for i in range(t, t + num_classes)]
        not_mask = np.setdiff1d(np.arange(len(outputs[0])), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        outputs[i] = outputs[i].index_fill(
            dim=0, index=not_mask, value=float(0) if zero else float('-inf'))

    return outputs

"""
def masking(outputs, labels, device, num_classes=2, mask_size=10, zero=False):
    classes_in_task = torch.unique(labels).tolist()
    mask = torch.ones(mask_size).to(device)
    mask = mask * -float('inf')
    mask[classes_in_task] = 0.

    return outputs + mask.unsqueeze(0)
"""

def get_cls_mask(labels, num_classes, mask_size, ctgy='current'):
    # mask_size: the number of total classes in all tasks
    classes_in_task = torch.unique(labels).tolist()
    unit = min(classes_in_task) // num_classes * num_classes

    classes_in_cur_task = torch.arange(unit, unit + num_classes).tolist()

    if ctgy == 'pre':
        classes_in_ctgy_task = torch.arange(unit).tolist()
    elif ctgy == 'sofar':
        classes_in_ctgy_task = torch.arange(unit + num_classes).tolist()
    else:  # current
        classes_in_ctgy_task = torch.arange(unit, unit + num_classes).tolist()

    cls_mask = torch.ones(mask_size)
    cls_mask = cls_mask * -float('inf')
    cls_mask[classes_in_ctgy_task] = 0.

    return cls_mask, classes_in_ctgy_task

