#! -*- coding:utf-8
from typing import Callable, List, Optional

import numpy as np
import torch
import torchvision

__all__ = ["CIFAR10", "FashionMNIST"]


class CIFAR10(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 indices: List[int] = None,
                 data_length: int = None,
                 shuffle: bool = False):
        super(CIFAR10, self).__init__()
        self.__datas__ = []
        self.__labels__ = []
        dataset = torchvision.datasets.CIFAR10(root,
                                               train=train,
                                               transform=transform,
                                               target_transform=target_transform,
                                               download=download)
        self.__classes__ = dataset.classes
        if indices is None:
            indices = list(range(len(dataset)))
        for i in indices:  # load data and catching...
            d, l = dataset[i]
            self.__datas__.append(d)
            self.__labels__.append(l)

        self.__length__ = (len(self.data)
                           if data_length is None else data_length)
        self.__indices__ = np.arange(len(self.data))
        self.__shuffle__ = shuffle
        if self.shuffle:
            np.random.shuffle(self.__indices__)
        self.__call_count__ = 0

    @property
    def data(self): return self.__datas__
    @property
    def label(self): return self.__labels__
    @property
    def classes(self): return self.__classes__
    @property
    def indices(self): return self.__indices__
    @property
    def shuffle(self): return self.__shuffle__
    def __len__(self): return self.__length__

    def __getitem__(self, idx):
        idx = self.indices[idx % len(self.data)]
        d = self.data[idx]
        l = self.label[idx]
        self.__call_count__ += 1
        if self.shuffle and self.__call_count__ >= len(self):
            np.random.shuffle(self.__indices__)
            self.__call_count__ = 0

        return d, l


class FashionMNIST(torch.utils.data.Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 indices: List[int] = None,
                 data_length: int = None,
                 shuffle: bool = False):
        super(FashionMNIST, self).__init__()
        self.__datas__ = []
        self.__labels__ = []
        dataset = torchvision.datasets.FashionMNIST(root,
                                                    train=train,
                                                    transform=transform,
                                                    target_transform=target_transform,
                                                    download=download)
        self.__classes__ = dataset.classes
        if indices is None:
            indices = list(range(len(dataset)))
        for i in indices:  # load data and catching...
            d, l = dataset[i]
            self.__datas__.append(d)
            self.__labels__.append(l)

        self.__length__ = (len(self.data)
                           if data_length is None else data_length)
        self.__indices__ = np.arange(len(self.data))
        self.__shuffle__ = shuffle
        if self.shuffle:
            np.random.shuffle(self.__indices__)
        self.__call_count__ = 0

    @property
    def data(self): return self.__datas__
    @property
    def label(self): return self.__labels__
    @property
    def classes(self): return self.__classes__
    @property
    def indices(self): return self.__indices__
    @property
    def shuffle(self): return self.__shuffle__
    def __len__(self): return self.__length__

    def __getitem__(self, idx):
        idx = self.indices[idx % len(self.data)]
        d = self.data[idx]
        l = self.label[idx]
        self.__call_count__ += 1
        if self.shuffle and self.__call_count__ >= len(self):
            np.random.shuffle(self.__indices__)
            self.__call_count__ = 0

        return d, l
