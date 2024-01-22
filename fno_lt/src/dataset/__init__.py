from .basic import CustomMNIST as MNIST
from .basic import CustomFMNIST as FashionMNIST
from .basic import CustomCIFAR10 as CIFAR10
from .basic import CustomCIFAR100 as CIFAR100
from .basic import ImageNet1k
from .colored_mnist import ColoredMNIST
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from .ina2018 import iNa2018
from .imbalance_imagenet import ImageNetLT
from .imbalance_places import PlacesLT
from .sampler import *


from torch.utils.data import Dataset

class MixIndexDataset(Dataset):
    def __init__(self, dataset):
        super(MixIndexDataset, self).__init__()
        self.dataset = dataset
        self.targets = dataset.targets

    def __getitem__(self, index):
        data_a, targets_a = self.dataset.__getitem__(index[0])
        data_b, targets_b = self.dataset.__getitem__(index[1])

        return (data_a, data_b), (targets_a, targets_b)

    def __len__(self):
        return len(self.dataset)

