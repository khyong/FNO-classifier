import os
import random, cv2
import numpy as np

from dataset.default import DefaultDataset


"""
class iNaturalist(DefaultDataset):
    def __init__(self, cfg, train=True, transform=None, **kwargs):
        super().__init__(cfg, train=train, transform=transform)
        seed_num = cfg.dataset.random_seed
        np.random.seed(seed_num)
        random.seed(seed_num)
        if self.dual_sample or (self.cfg.train.sampler.type == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        img_info = self.data[index]
        img = self._get_image(img_info)
        image = self.transform(img)

        meta = dict()
        if self.dual_sample:
            if self.cfg.train.sampler.dual_sampler.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
            elif self.cfg.train.sampler.dual_sampler.type == "balance":
                sample_class = random.randint(0, self.num_classes - 1)

            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_info = self.data[sample_index]
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id']
            sample_img = self.transform(sample_img)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.train:
            image_label = img_info['category_id']  # 0-index

        return image, image_label, meta
"""
class iNaturalist(DefaultDataset):
    def __init__(self, cfg, train=True, transform=None, **kwargs):
        super().__init__(cfg, train=train, transform=transform)
        
