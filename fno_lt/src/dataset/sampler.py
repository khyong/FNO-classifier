# This code is copied from the official PyTorch implementation in MiSLAS
# (https://github.com/dvlab-research/MiSLAS)
import math
import random
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data.distributed import Sampler, Dataset

from typing import TypeVar, Optional, Iterator

T_co = TypeVar('T_co', covariant=True)

__all__ = [
    'BalancedDatasetSampler',
    'EffectNumSampler',
    'RandomCycleIter',
    'ClassAwareSampler',
    'DistributedClassAwareSampler',
    'DistributedMixIndexSampler',
]


class BalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
        per_cls_weights = 1 / np.array(label_to_count)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


class EffectNumSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.targets[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples


class RandomCycleIter:

    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode
        
    def __iter__ (self):
        return self
    
    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_per_cls=1):

    i = 0
    j = 0
    while i < n:
        if j >= num_samples_per_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_per_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1

class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_per_cls=4,):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_per_cls = num_samples_per_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_per_cls)
    
    def __len__ (self):
        return self.num_samples
 

def get_sampler():
    return ClassAwareSampler


class DistributedClassAwareSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, data_source, num_samples_per_cls=4,):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
#        self.num_samples = int(self.num_samples / 4)
        self.num_samples_per_cls = num_samples_per_cls
        
    def __iter__ (self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_per_cls)
    
    def __len__ (self):
        return self.num_samples


class DistributedMixIndexSampler(Sampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 mix_index_func = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.mix_index_func = mix_index_func

    def __iter__(self) -> Iterator[T_co]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.mix_index_func(len(self.dataset), self.dataset.targets)
        if isinstance(indices, tuple):
            indices_a, indices_b = indices[0].tolist(), indices[1].tolist()
        else:
            indices_a = indices.tolist()
            indices_b = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]

        if not self.drop_last:
            padding_size = self.total_size - len(indices_a)
            if padding_size <= len(indices_a):
                indices_a += indices_a[:padding_size]
                indices_b += indices_b[:padding_size]
            else:
                indices_a += (indices_a * math.ceil(padding_size / len(indices_a)))[:padding_size]
                indices_b += (indices_b * math.ceil(padding_size / len(indices_b)))[:padding_size]
        else:
            indices_a = indices_a[:self.total_size]
            indices_b = indices_b[:self.total_size]

        # subsample
        indices_a = indices_a[self.rank:self.total_size:self.num_replicas]
        indices_b = indices_b[self.rank:self.total_size:self.num_replicas]

        return iter([[idx_a, idx_b] for idx_a, idx_b in zip(indices_a, indices_b)])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

