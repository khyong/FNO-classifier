# The ImageNetLT class is based on the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
import os
import random
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms

from dataset.default import DefaultDataset


class iNa2018(DefaultDataset):
    num_classes = 8142

    def __init__(self, cfg, train=True, class_map=None, **kwargs):
        super(iNa2018, self).__init__(cfg, train=train)
        seed_num = cfg.dataset.random_seed
        
        normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
        if self.train:
            np.random.seed(seed_num)
            random.seed(seed_num)
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
                transforms.ToTensor(),
                normalize,
            ])
            self.txt = cfg.dataset.train_info
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            self.txt = cfg.dataset.valid_info

        self.img_path = []
        self.targets = []
        with open(self.txt) as f:
            for line in f:
                self.img_path.append(os.path.join(cfg.dataset.root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        if self.train and (class_map is None):
            self.img_num_per_cls, self.class_map, self.targets = self.get_img_num_per_cls()
        else:
            self.class_map = class_map
            self.targets = np.array(self.class_map)[self.targets].tolist()

    def get_img_num_per_cls(self):
        img_num_per_cls_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(img_num_per_cls_old))
        class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            class_map[sorted_classes[i]] = i

        targets = np.array(class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(targets)):
            j = targets[i]
            self.class_data[j].append(i)

        img_num_per_cls = [np.sum(np.array(targets)==i) for i in range(self.num_classes)]
        return img_num_per_cls, class_map, targets
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target 

