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


class PlacesLT(DefaultDataset):
    num_classes = 365

    def __init__(self, cfg, train=True, class_map=None, **kwargs):
        super(PlacesLT, self).__init__(cfg, train=train)
        seed_num = cfg.dataset.random_seed

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            self.txt = os.path.join(cfg.dataset.root, 'data_txt/Places_LT_train.txt')
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            self.txt = os.path.join(cfg.dataset.root, 'data_txt/Places_LT_test.txt')

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

        mode = 'train' if self.train else 'valid'
        print("{} Mode: Contain {} images".format(mode, len(self.targets)))
        if self.dual_sample or (self.cfg.train.sampler.type == 'weighted_sampler' and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.num_classes)
            self.class_dict = self._get_class_dict()
        
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
        meta = dict()

        if self.dual_sample:
            if self.cfg.train.sampler.dual_sampler.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.train.sampler.dual_sampler.type == "balance":
                sample_class = random.randint(0, self.num_classes-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.train.sampler.dual_sampler.type == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.data[sample_index], self.targets[sample_index]
            sample_img = Image.fromarray(sample_img)
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        with open(path, 'rb') as f:
            sample = Image.open(f).convert(self.cfg.color_space)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, meta

    def get_annotations(self):
        annots = []
        for target in self.targets:
            annots.append({'category_id': int(target)})
        return annots

    def _get_class_dict(self):
        class_dict = dict()
        annots = self.get_annotations()
        for i, annot in enumerate(annots):
            ctgy_id = annot['category_id']
            if not ctgy_id in class_dict:
                class_dict[ctgy_id] = []
            class_dict[ctgy_id].append(i)
        return class_dict

