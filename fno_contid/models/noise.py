# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser

from models.utils import num_classes, mask_size, masking


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Noise(ContinualModel):
    NAME = 'noise'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Noise, self).__init__(backbone, loss, args, transform)
        if args.dataset not in num_classes:
            raise NotImplementedError("Update num_classes")
        if args.dataset not in mask_size:
            raise NotImplementedError("Update mask_size")
        self.num_classes = num_classes[args.dataset]
        self.mask_size = mask_size[args.dataset]

    def observe(self, inputs, labels, not_aug_inputs):
        #gaussian_noise = torch.normal(mean=0., std=torch.ones_like(inputs)*1e-3).to(self.device)
        v_noise = torch.from_numpy(np.random.beta(0.1, 0.1, size=inputs.shape)).float().to(self.device)
        v_noise = 1.0 * (v_noise - 0.5)
        noise_inputs = v_noise + inputs.detach().clone()

        self.opt.zero_grad()
        outputs = self.net(inputs)
        masked_outputs = masking(
            outputs, labels, self.device,
            num_classes=self.num_classes,
            mask_size=self.mask_size,
            zero=False)
        loss = self.loss(masked_outputs, labels)
    
        start_cls_num = (torch.min(labels) // self.num_classes) * self.num_classes
        #cur_classes = [i for i in range(start_cls_num, start_cls_num + self.num_classes)]
        sofar_classes = [i for i in range(start_cls_num + self.num_classes)]

        noise_outputs = self.net(noise_inputs)
        noise_outputs[:,sofar_classes] = -float('inf')

        for cls_num in range(self.mask_size):
            #if cls_num in cur_classes: continue
            if cls_num in sofar_classes: continue
            tmp_labels = cls_num * torch.ones_like(labels).to(self.device)
            loss += (1. / (self.mask_size - self.num_classes)) * self.loss(noise_outputs, tmp_labels)

        loss.backward()
        self.opt.step()

        return loss.item()
