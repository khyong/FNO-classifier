import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image


class ColoredMNIST(MNIST):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        palette='rgb',
        target_type='digit',
        biased=False,
        **kwargs,
    ):
        super(ColoredMNIST, self).__init__(cfg.dataset.root, train, transform, target_transform, download)
        self.num_classes = len(self.classes)
        self.target_type = target_type

        data, targets, targets_color = self.colorized(
            self.data, self.targets, palette=palette, biased=biased)

        if target_type == 'color':
            self.data, self.targets = data, targets_color
        elif target_type == 'both':
            self.data, self.targets_digit, self.targets_color = data, targets, targets_color
        else:
            self.data, self.targets = data, targets

    def colorized(self, data, targets, palette='gray', biased=False):
        if palette == 'rgb':
            color_dict = {0: 'red', 1: 'green', 2: 'blue'}
            mask = torch.eye(3, 3, dtype=torch.long)
        elif palette == 'all':
            color_dict = {
                0: 'red', 1: 'green', 2: 'blue', 
                3: 'yellow', 4: 'pink', 5: 'cyan',
            }
            mask = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1]
            ], dtype=torch.long)
        else:
            return data, targets, None
        
        num_colors = len(color_dict)
        num_samples_per_color = [len(data)//num_colors] * num_colors
        num_samples_per_color[0] += len(data)%num_colors
                   
        data = torch.stack([data] * 3, dim=1)
        targets =  targets.numpy()
        if biased and self.train and (palette == 'rgb'):
            targets_color = np.zeros(targets.shape, dtype=np.int64)
            for i in range(self.num_classes):
                if 3 <= i < 7:
                    targets_color[np.where(targets==i)] = 1
                elif i >= 7:
                    targets_color[np.where(targets==i)] = 2
        else:
            targets_color = np.random.permutation(
                np.concatenate([
                    [i]*num_samples_per_color[i] 
                    for i in range(num_colors)
                ])
            )
            
        for i in range(num_colors):
            data[np.where(targets_color==i)[0]] *= mask[i].reshape(1, -1, 1, 1)
        
        return data.permute(0, 2, 3, 1), torch.from_numpy(targets), torch.from_numpy(targets_color)

    def __getitem__(self, index):
        if self.target_type == 'both':
            img, target_d, target_c = self.data[index], self.targets_digit[index], self.targets_color[index]
            img = Image.fromarray(img.numpy())

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target_d = self.target_transform(target_d)
                target_c = self.target_transform(target_c)

            return img, target_d, target_c
        
        else:
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img.numpy())

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
