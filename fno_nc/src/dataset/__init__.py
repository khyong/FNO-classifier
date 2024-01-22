from .basic import CustomMNIST as MNIST
from .basic import CustomSVHN as SVHN
from .basic import CustomFMNIST as FashionMNIST
from .basic import CustomCIFAR10 as CIFAR10
from .basic import CustomCIFAR100 as CIFAR100
from .basic import CustomSTL10 as STL10
from .basic import ImageNet1k
from .basic import BalancedMNIST, BalancedSVHN, BalancedImageNet1k
from .colored_mnist import ColoredMNIST
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from .iNaturalist import iNaturalist
from .imbalance_imagenet import ImageNetLT
from .places import PlacesLT
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

