#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:24:22 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import torch.nn as nn


class MNISTMLP(nn.Module):
    """
    MLP for MNIST
    """
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, return_intermediate=False):
        all_x = []
        x = x.view(-1, 28 * 28)
        if return_intermediate:
            all_x.append(x)
        x = self.fc1(x)
        if return_intermediate:
            all_x.append(x)
        x = self.fc2(x)
        if return_intermediate:
            all_x.append(x)
        x = self.fc3(x)
        if return_intermediate:
            all_x.append(x)
        x = self.soft(x)
        if return_intermediate:
            return x, all_x
        else:
            return x
