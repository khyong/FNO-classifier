# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    num_classes = 10

    def __init__(self, cfg, train=True,
                 transform=None, target_transform=None, download=True,
                 **kwargs):
        super(IMBALANCECIFAR10, self).__init__(cfg.dataset.root, train, transform, target_transform, download)
        self.cfg = cfg
        self.train = train
        self.dual_sample = True if cfg.train.sampler.dual_sampler.enable and self.train else False
        seed_num = cfg.dataset.random_seed
        if self.train:
            np.random.seed(seed_num)
            random.seed(seed_num)
            imb_factor = self.cfg.dataset.imbalancecifar.ratio
            img_num_list = self.get_img_num_per_cls(self.num_classes, cfg.dataset.imbalancecifar.imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        mode = 'train' if self.train else 'valid'
        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.train.sampler.type == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.num_classes)
            self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, num_classes, imb_type, imb_factor):
        img_max = len(self.data) / num_classes
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(num_classes):
                num = img_max * (imb_factor**(cls_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * num_classes)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        v_rand, v_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            v_sum += self.class_weight[i]
            if v_rand <= v_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, annot in enumerate(self.get_annotations()):
            ctgy_id = annot["category_id"]
            if not ctgy_id in class_dict:
                class_dict[ctgy_id] = []
            class_dict[ctgy_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        ctgy_list = []
        for annot in annotations:
            ctgy_id = annot["category_id"]
            num_list[ctgy_id] += 1
            ctgy_list.append(ctgy_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.cfg.train.sampler.type == "weighted sampler" and self.train:
            assert self.cfg.train.sampler.weighted_sampler.type in ["balance", "reverse"]
            if  self.cfg.train.sampler.weighted_sampler.type == "balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.cfg.train.sampler.weighted_sampler.type == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img, target = self.data[index], self.targets[index]
        #meta = dict()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

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

            #meta['sample_image'] = sample_img
            #meta['sample_label'] = sample_label

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, meta
        return img, target

    def get_num_classes(self):
        return self.num_classes

    def reset_epoch(self, epoch):
        self.epoch = epoch

    def get_annotations(self):
        annots = []
        for target in self.targets:
            annots.append({'category_id': int(target)})
        return annots

    def gen_imbalanced_data(self, img_num_per_cls):
        data, targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            data.append(self.data[selec_idx, ...])
            targets.extend([the_class, ] * the_img_num)
        data = np.vstack(data)
        self.data = data
        self.targets = targets
        

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
