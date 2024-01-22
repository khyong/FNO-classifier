from torchvision.datasets import MNIST, SVHN, FashionMNIST, CIFAR10, CIFAR100, STL10


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


class CustomSVHN(SVHN):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        self.__class__.__name__ = 'SVHN'
        split = 'train' if train else 'test'
        super(CustomSVHN, self).__init__(cfg.dataset.root, split, transform, target_transform, download)
        self.targets = self.labels


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


class CustomSTL10(STL10):
    def __init__(
        self, 
        cfg,
        train=True,
        folds=None,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        split = 'train' if train else 'test'
        super(CustomSTL10, self).__init__(cfg.dataset.root, split, folds, transform, target_transform, download)
        self.targets = self.labels


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


def balanced_sampling(targets, num=600):
    import torch
    import numpy as np
 
    total_indices = torch.arange(len(targets))

    old_targets = targets if isinstance(targets, torch.Tensor) else torch.tensor(targets)
    
    tgt, cnt = torch.unique(old_targets, return_counts=True)
    new_indices = []
    for i, c in zip(tgt.tolist(), cnt.tolist()):
        indices = np.random.permutation(np.arange(c))
        new_indices.append(total_indices[old_targets==i][indices[:num]])

    return np.concatenate(new_indices)


class BalancedMNIST(MNIST):
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
        super(BalancedMNIST, self).__init__(cfg.dataset.root, train, transform, target_transform, download)

        balanced_indices = balanced_sampling(self.targets, num=5000)
        self.data, self.targets = self.data[balanced_indices], self.targets[balanced_indices]
        

class BalancedSVHN(SVHN):
    def __init__(
        self, 
        cfg,
        train=True,
        transform=None, 
        target_transform=None, 
        download=False,
        **kwargs,
    ):
        self.__class__.__name__ = 'SVHN'
        split = 'train' if train else 'test'
        super(BalancedSVHN, self).__init__(cfg.dataset.root, split, transform, target_transform, download)

        balanced_indices = balanced_sampling(self.labels, num=4600)
        self.data, self.labels = self.data[balanced_indices], self.labels[balanced_indices]
        self.targets = self.labels


class BalancedImageNet1k(ImageFolder):
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
        
        balanced_indices = balanced_sampling(self.targets, num=600)
        new_samples = []
        for i, s in enumerate(samples):
            if i in balanced_indices:
                new_samples.append(s)
        
        self.samples = new_samples
        self.targets = [s[1] for s in new_samples]

