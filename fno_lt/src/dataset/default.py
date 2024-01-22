import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import os
import cv2
import json
import time
import random
import numpy as np

from data_transform.transform_wrapper import TRANSFORMS


class DefaultDataset(Dataset):
    def __init__(self, cfg, train=True, transform=None):
        self.cfg = cfg
        self.train = train
        self.transform = transform

        self.input_size = cfg.input_size
        self.data_type = cfg.dataset.data_type
        self.color_space = cfg.color_space
        self.size = self.input_size
        self.dual_sample = True if cfg.train.sampler.dual_sampler.enable and self.train else False

        print("Use {} Mode to train network".format(self.color_space))
        if self.data_type != 'nori':
            self.data_root = cfg.dataset.root
        else:
            self.fetcher = None

        if self.train:
            print("Loading train data ...", end=" ")
            self.info_path = cfg.dataset.train_info
        else:
            print("Loading valid data ...", end=" ")
            self.info_path = cfg.dataset.valid_info

        self.update_transform()

        if self.info_path != "":
            _, ext = os.path.splitext(self.info_path)
            if ext == '.json':
                with open(self.info_path, 'r') as f:
                    self.all_info = json.load(f)
                self.num_classes = self.all_info['num_classes']
                self.data = self.all_info['annotations']
                print("Contain {} images of {} classes".format(len(self.data), self.num_classes))

    def __getitem__(self, index):
        img_info = self.data[index]
        img = self._get_image(img_info)
#        meta = dict()
        image = self.transform(img)
        label = (img_info['category_id'] if self.train else 0)  # 0-dinex
#        if self.train is None:
#            meta['image_id'] = img_info['image_id']
#            meta['fpath'] = img_info['fpath']

#        return image, label, meta
        return image, label

    def update_transform(self, input_size=None):
        transform_ops = (
            self.cfg.transforms.train_transforms
            if self.train
            else self.cfg.transforms.test_transforms
        )

        transform_list = [transforms.ToPILImage()]
        for op in transform_ops:
            if op == 'normalize': continue
            transform_list.append(TRANSFORMS[op](cfg=self.cfg, input_size=input_size))
        transform_list.append(transforms.ToTensor())

        if 'normalize' in transform_ops:
            transform_list.append(TRANSFORMS['normalize'](self.cfg))

        self.transform = transforms.Compose(transform_list)

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, img_info):
        if self.data_type == 'jpg':
            fpath = os.path.join(self.data_root, img_info['fpath'])
            img = self.imread_with_retry(fpath)
        if self.color_space == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, annot in enumerate(self.data):
            ctgy_id = (
                annot['category_id'] if 'category_id' in annot
                else annot['image_label']
            )
            if not ctgy_id in class_dict:
                class_dict[ctgy_id] = []
            class_dict[ctgy_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        ctgy_list = []
        for annot in annotations:
            ctgy_id = annot['category_id']
            num_list[ctgy_id] += 1
            ctgy_list.append(ctgy_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        v_rand, v_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            v_sum += self.class_weight[i]
            if v_rand <= v_sum:
                return i
            
