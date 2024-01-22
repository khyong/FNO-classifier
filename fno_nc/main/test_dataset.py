import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import os
import sys
import ast
import click
import shutil
import random
import argparse
import warnings
import numpy as np

import _init_paths
import loss as custom_loss
import dataset as custom_dataset
from data_transform.transform_wrapper import get_transform
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
)
from core.function import train_model, valid_model, test_model
from core.trainer import Trainer
from utils.reprod import fix_seed
from utils.dist import setup, cleanup


def parse_args():
    parser = argparse.ArgumentParser(description="Codes for Orth")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def main_worker(rank, world_size, args):
    # ----- BEGIN basic setting -----
    update_config(cfg, args)
    logger = None

    fix_seed(cfg.seed_num)

    torch.cuda.set_device(rank)
    # ----- END basic setting -----

    # ----- BEGIN dataset setting -----
    transform_tr = get_transform(cfg, mode='train')
    transform_ts = get_transform(cfg, mode='test')

    train_set = getattr(custom_dataset, cfg.dataset.dataset)(
        cfg, train=True, download=True, transform=transform_tr)

    data = train_set.data if isinstance(train_set.data, torch.Tensor) else torch.tensor(train_set.data)
    targets = train_set.targets if isinstance(train_set.targets, torch.Tensor) else torch.tensor(train_set.targets)

    print(torch.unique(targets, return_counts=True))
    print(data.shape)
    import torchvision.transforms as transforms
    from PIL import Image
    data_tensor = []
    tt = transforms.ToTensor()

    if cfg.dataset.dataset in ['STL10', 'BalancedSVHN']:
        data = data.permute(0, 2, 3, 1)
    for img in data:
        orig_img = Image.fromarray(img.numpy(), mode='L' if data.dim() == 3 else 'RGB')
        data_tensor.append(tt(orig_img).unsqueeze(0))
    data_tensor = torch.vstack(data_tensor)
    print(data_tensor.shape)
    if data_tensor.dim() == 3:
        print("mean: ", torch.mean(data_tensor))
        print("std : ", torch.std(data_tensor))
    else:
        print("mean: ", torch.mean(data_tensor.permute(1, 0, 2, 3).flatten(1), dim=1))
        print("std : ", torch.std(data_tensor.permute(1, 0, 2, 3).flatten(1), dim=1))

if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    rank = cfg.rank if cfg.rank != -1 else 0
    main_worker(rank, cfg.world_size, args)

