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
from utils.cfsgm import cfsgm

from models.utils import num_classes, mask_size, get_cls_mask, masking


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Masked Replay + CFSGM.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class SGDCFSGM(ContinualModel):
    NAME = 'sgdcfsgm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SGDCFSGM, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        if args.dataset not in num_classes:
            raise NotImplementedError("Update num_classes")
        self.num_classes = num_classes[args.dataset]
        if args.dataset not in mask_size:
            raise NotImplementedError("Update mask_size")
        self.mask_size = mask_size[args.dataset]
        self.alpha = args.nomemalpha
        print("ALPHA: ", self.alpha)

    """
    def observe(self, inputs, labels, not_aug_inputs, pre_net=None):
        # 1. masking - mask before softmax
        cur_mask, classes_in_cur_task = get_cls_mask(
            labels, self.num_classes, self.mask_size, ctgy='current')
        cur_mask = cur_mask.to(self.device)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        masked_outputs = cur_mask.unsqueeze(0) + outputs
        loss = self.loss(masked_outputs, labels)
        
        loss.backward(retain_graph=True)
        self.opt.step()

        if min(classes_in_cur_task) // self.num_classes > 0 and pre_net is not None:
            pre_net = pre_net.to(self.device)
            pre_net.eval()
            self.opt.zero_grad()

            loss = 0.
            _, classes_in_pre_task = get_cls_mask(
                labels, self.num_classes, self.mask_size, ctgy='pre')
            if len(classes_in_pre_task) != 0:
                cls_num_list = np.random.choice(classes_in_pre_task, inputs.shape[0])
                fake_labels = torch.from_numpy(cls_num_list).long()
                adv_images, adv_labels, _ = cfsgm(
                    inputs, fake_labels, self.net, self.loss, self.device, alpha=self.alpha)
                with torch.no_grad():
                    adv_outputs = pre_net(adv_images)
                cur_adv_outputs = self.net(adv_images)
                loss += F.mse_loss(adv_outputs, cur_adv_outputs)

            loss.backward()
            self.opt.step()

        return loss.item()
    """
    """ minchan masking """
    def observe(self, inputs, labels, not_aug_inputs, pre_net=None):
        # 1. masking - mask before softmax
        _, classes_in_cur_task = get_cls_mask(
            labels, self.num_classes, self.mask_size, ctgy='current')

        self.opt.zero_grad()
        outputs = self.net(inputs)
        masked_outputs = masking(
            outputs, labels, self.device,
            num_classes=self.num_classes,
            mask_size=self.mask_size,
            zero=False)
        loss = self.loss(masked_outputs, labels)
        
        loss.backward(retain_graph=True)
        self.opt.step()

        if min(classes_in_cur_task) // self.num_classes > 0 and pre_net is not None:
            pre_net = pre_net.to(self.device)
            pre_net.eval()
            self.opt.zero_grad()

            loss = 0.
            _, classes_in_pre_task = get_cls_mask(
                labels, self.num_classes, self.mask_size, ctgy='pre')
            if len(classes_in_pre_task) != 0:
                cls_num_list = np.random.choice(classes_in_pre_task, inputs.shape[0])
                fake_labels = torch.from_numpy(cls_num_list).long()
                adv_images, adv_labels, _ = cfsgm(
                    inputs, fake_labels, self.net, self.loss, self.device, alpha=self.alpha)
                with torch.no_grad():
                    adv_outputs = pre_net(adv_images)
                cur_adv_outputs = self.net(adv_images)
                loss += F.mse_loss(adv_outputs, cur_adv_outputs)

            loss.backward()
            self.opt.step()

        return loss.item()

