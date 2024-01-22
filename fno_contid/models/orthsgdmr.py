# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser

from models.utils import num_classes, mask_size, masking


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class OrthSgdMR(ContinualModel):
    NAME = 'orthsgdmr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(OrthSgdMR, self).__init__(backbone, loss, args, transform)
        if args.dataset not in num_classes:
            raise NotImplementedError("Update num_classes")
        if args.dataset not in mask_size:
            raise NotImplementedError("Update mask_size")
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
        loss.backward()
        self.opt.step()

        return loss.item()
