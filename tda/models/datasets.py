#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:27:17 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from operator import itemgetter
from random import shuffle, seed

from tda.logging import get_logger

_root = './data'
_trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

torch.manual_seed(1)
seed(1)

logger = get_logger("Datasets")


class Dataset(object):

    _datasets = dict()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @classmethod
    def get_or_create(
            cls,
            name: str,
            validation_size: int = 1000
    ):
        """
        Using singletons for dataset to shuffle once at the beginning of the run
        """
        if (name, validation_size) not in Dataset._datasets:
            Dataset._datasets[(name, validation_size)] = cls(
                name=name,
                validation_size=validation_size
            )
            logger.info(f"Instantiated dataset {name} with validation_size {validation_size}")
        return Dataset._datasets[(name, validation_size)]

    def __init__(self,
                 name: str,
                 validation_size: int = 1000
                 ):

        self.name = name.lower()

        if name == "MNIST":
            self.train_dataset = dset.MNIST(
                root=_root,
                train=True,
                transform=_trans,
                download=True)

            self.test_and_val_dataset = dset.MNIST(
                root=_root,
                train=False,
                transform=_trans,
                download=True)
        elif name == "SVHN":
            self.train_dataset = dset.SVHN(
                root=_root,
                split="train",
                transform=_trans,
                download=True)

            self.test_and_val_dataset = dset.SVHN(
                root=_root,
                split="test",
                transform=_trans,
                download=True)
        elif name == "CIFAR10":
            self.train_dataset = dset.CIFAR10(
                root=_root,
                train=True,
                transform=_trans,
                download=True)

            self.test_and_val_dataset = dset.CIFAR10(
                root=_root,
                train=False,
                transform=_trans,
                download=True)
        else:
            raise NotImplementedError(
                f"Unknown dataset {name}"
            )

        self.val_dataset = list()
        self.test_dataset = list()
        for i, x in enumerate(self.test_and_val_dataset):
            if i < validation_size:
                self.val_dataset.append(x)
            else:
                self.test_dataset.append(x)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=100,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            shuffle=True,
            batch_size=1)
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True)

        self.train_dataset = list(self.train_dataset)
        self.test_and_val_dataset = list(self.test_and_val_dataset)
        shuffle(self.train_dataset)
        shuffle(self.test_and_val_dataset)

        self.test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
                                                                           self.test_loader.dataset)),
                                             map(itemgetter(1), self.test_loader.dataset)))
