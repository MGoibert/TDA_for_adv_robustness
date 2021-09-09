#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:27:17 2019
"""

from random import shuffle, seed

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn import datasets
import numpy as np

from tda.tda_logging import get_logger
from tda.devices import device
from tda.dataset.tiny_image_net import load_tiny_image_net
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)

_root = "./data"
_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)
_trans_BandW = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

torch.manual_seed(1)
seed(1)

logger = get_logger("Datasets")

class dsetsCircleToy(torch.utils.data.Dataset):
    def __init__(self, n_samples=5000, noise=0.05, factor=0.5):
        X_, Y_ = datasets.make_circles(n_samples=n_samples, shuffle=True,
            noise=noise, factor=factor)
        X_ = [(x_ + 1.3)/2.6 for x_ in X_]
        Y__ = np.reshape(Y_, len(Y_))
        self.X = torch.tensor(X_, dtype=torch.float)
        self.Y = torch.tensor(Y__, dtype=torch.long)
        self.n_samples=n_samples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.X[idx,:]
        y = self.Y[idx]
        return [x, y]

class dsetsViz(torch.utils.data.Dataset):
    def __init__(self, n_samples=5000):
        r = np.random.permutation(3*n_samples)
        x0 = [1]*3 + [0]*3 + [0]*3
        x1 = [0]*3 + [1]*3 + [0]*3
        x2 = [0]*3 + [0]*3 + [1]*3
        x_ = [[x0]*n_samples + [x1]*n_samples + [x2]*n_samples]
        x_ = np.asarray(x_)
        y_ = np.asarray([0]*n_samples+[1]*n_samples+[2]*n_samples)
        np.take(x_, r, axis=1,out=x_)
        np.take(y_, r, axis=0,out=y_)
        self.X = torch.tensor(x_, dtype=torch.float).squeeze(0)
        self.Y = torch.tensor(y_, dtype=torch.long)
        self.n_samples = n_samples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.X[idx,:]
        y = self.Y[idx]
        return [x, y]


class Dataset(object):

    _datasets = dict()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create(cls, name: str, validation_size: int = 1000):
        """
        Using singletons for dataset to shuffle once at the beginning of the run
        """
        if (name, validation_size) not in Dataset._datasets:
            Dataset._datasets[(name, validation_size)] = cls(
                name=name, validation_size=validation_size
            )
            logger.info(
                f"Instantiated dataset {name} with validation_size {validation_size}"
            )
        return Dataset._datasets[(name, validation_size)]

    def __init__(self, name: str, validation_size: int = 1000):

        self.name = name.lower()

        if name == "MNIST":
            self.train_dataset = dset.MNIST(
                root=_root, train=True, transform=_trans, download=True
            )
            logger.info(f"MNIST {self.train_dataset}")

            self.test_and_val_dataset = dset.MNIST(
                root=_root, train=False, transform=_trans, download=True
            )
        elif name == "SVHN":
            self.train_dataset = dset.SVHN(
                root=_root, split="train", transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.SVHN(
                root=_root, split="test", transform=_trans, download=True
            )
        elif name == "SVHN_BandW":
            self.train_dataset = dset.SVHN(
                root=_root, split="train", transform=_trans_BandW, download=True
            )

            self.test_and_val_dataset = dset.SVHN(
                root=_root, split="test", transform=_trans_BandW, download=True
            )
        elif name == "CIFAR10":
            self.train_dataset = dset.CIFAR10(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.CIFAR10(
                root=_root, train=False, transform=_trans, download=True
            )
        elif name == "FashionMNIST":
            self.train_dataset = dset.FashionMNIST(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.FashionMNIST(
                root=_root, train=False, transform=_trans, download=True
            )
        elif name == "CircleToy":
            self.train_dataset = dsetsCircleToy(2000)
            self.test_and_val_dataset = dsetsCircleToy(2000)

        elif name == "TinyImageNet":
            self.train_dataset = load_tiny_image_net(mode="train", transform=_trans)
            self.test_and_val_dataset = load_tiny_image_net(mode="test", transform=_trans)

        elif name == "ToyViz":
            self.train_dataset = dsetsViz(1000)
            self.test_and_val_dataset = dsetsViz(1000)

        elif name == "CIFAR100":
            self.train_dataset = dset.CIFAR100(
                root=_root, train=True, transform=_trans, download=True
            )

            self.test_and_val_dataset = dset.CIFAR100(
                root=_root, train=False, transform=_trans, download=True
            )
            
        else:
            raise NotImplementedError(f"Unknown dataset {name}")

        self.val_dataset = list()
        self.test_dataset = list()
        for i, x in enumerate(self.test_and_val_dataset):
            if i < validation_size:
                self.val_dataset.append(x)
            else:
                self.test_dataset.append(x)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=128, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, shuffle=True, batch_size=2048
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=128, shuffle=True
        )

        self.train_dataset = list(self.train_dataset)
        self.test_and_val_dataset = list(self.test_and_val_dataset)
        shuffle(self.train_dataset)
        shuffle(self.test_and_val_dataset)

        # self.test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
        #                                                                   self.test_loader.dataset)),
        #                                     map(itemgetter(1), self.test_loader.dataset)))
