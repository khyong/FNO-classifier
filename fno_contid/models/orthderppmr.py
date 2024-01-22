# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer

from models.utils import num_classes, mask_size, masking


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Fixed Orthnormal Linear.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class OrthDerppMR(ContinualModel):
    NAME = 'orthderppmr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(OrthDerppMR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if args.dataset not in num_classes:
            raise NotImplementedError("Update num_classes")
        self.num_classes = num_classes[args.dataset]
        self.mask_size = mask_size[args.dataset]

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        masked_outputs = masking(
            outputs, labels, self.device,
            num_classes=self.num_classes,
            mask_size=self.mask_size,
            zero=False)
        loss = self.loss(masked_outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

        outputs_ = self.net(inputs)
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs_.data)

        return loss.item()
 
