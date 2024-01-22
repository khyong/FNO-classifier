from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100


class CustomMNIST(MNIST):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        self.__class__.__name__ = 'MNIST'
        super(CustomMNIST, self).__init__(cfg.dataset.root, train, transform, target_transform, download)


class CustomFMNIST(FashionMNIST):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        self.__class__.__name__ = 'FashionMNIST'
        super(CustomFMNIST, self).__init__(cfg.dataset.root, train, transform, target_transform, download)


class CustomCIFAR10(CIFAR10):
    base_folder = "cifar-10-batches-py"

    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        super(CustomCIFAR10, self).__init__(cfg.dataset.root, train, transform, target_transform, download)


class CustomCIFAR100(CIFAR100):
    base_folder = "cifar-100-python"

    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        super(CustomCIFAR100, self).__init__(cfg.dataset.root, train, transform, target_transform, download)


from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import os


class ImageNet1k(ImageFolder):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        **kwargs,
    ):
        tag = 'train' if train else 'val'
        datadir = os.path.join(cfg.dataset.root, tag)
        super().__init__(
            datadir,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )

