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

# MNIST dataset
root = './data'
trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True,
                           transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False,
                          transform=trans, download=True)

val_data = []
test = []
for i, x in enumerate(test_set):
    if i < 1000:
        val_data.append(x)
    else:
        test.append(x)

lims = -0.5, 0.5

train_loader_MNIST = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=100, shuffle=True)
test_loader_MNIST = torch.utils.data.DataLoader(dataset=test, shuffle=True,
                                          batch_size=1)
val_loader_MNIST = torch.utils.data.DataLoader(dataset=val_data, batch_size=len(val_data),
                                         shuffle=True)
test_loader_MNIST.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
            test_loader_MNIST.dataset)), map(itemgetter(1), test_loader_MNIST.dataset)))