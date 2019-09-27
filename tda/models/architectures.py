#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:24:22 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable
from functools import reduce
from scipy.linalg import toeplitz

torch.set_default_tensor_type(torch.DoubleTensor)

##########
# Layers #
##########


class Layer(object):

    def __init__(self,
                 func: nn.Module,
                 graph_layer: bool
                 ):
        self.func = func
        self.graph_layer = graph_layer
        self._activations = None

    def get_matrix(self):
        raise NotImplementedError()

    def process(self, x, store_for_graph):
        if store_for_graph:
            self._activations = x
        return self.func(x)


class LinearLayer(Layer):

    def __init__(self, in_width, out_width):
        super().__init__(
            func=nn.Linear(in_width, out_width),
            graph_layer=True
        )

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        m = list(self.func.parameters())[0]
        return np.abs((self._activations * m).detach().numpy())


class ConvLayer(Layer):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            func=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False
            ),
            graph_layer=True
        )

    @staticmethod
    def _get_nb_elements(t):
        """
        Return the number of element of a given tensor
        """
        return reduce(lambda a, b: a * b, list(t.shape))

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        print(f"My activations : {self._activations}")
        for param in self.func.parameters():
            kernel = torch.squeeze(param.data)
        print(f"My kernel: {kernel}")

        # 1) Compute the size of the matrix
        nbcols = ConvLayer._get_nb_elements(self._activations)
        print(f"NbCols={nbcols}")

        kernel_size = ConvLayer._get_nb_elements(kernel)
        print(f"Kernel_size={kernel_size}")

        out = self.func(self._activations)
        nbrows = ConvLayer._get_nb_elements(out)
        print(f"NbRows={nbrows}")

        # 2) Compute size of the Toeplitz matrix

        nbrows_t = list(self._activations.shape)[-1] - list(kernel.shape)[-1] + 1
        nbcols_t = list(self._activations.shape)[-1]
        print(f"The Toeplitz matrices are {nbrows_t}x{nbcols_t}")

        zero_toeplitz = np.zeros((nbrows_t, nbcols_t))

        toeplitz_matrices = list()
        for i in range(list(kernel.shape)[-2]):
            row = kernel.detach().numpy()[i, :]
            row_toeplitz = np.zeros((1, nbcols_t))
            row_toeplitz[0, :row.shape[0]] = row
            col_toeplitz = np.zeros((1, nbrows_t))
            col_toeplitz[0, 0] = row_toeplitz[0, 0]
            topl = toeplitz(col_toeplitz, row_toeplitz)
            toeplitz_matrices.append(topl)

        nb_blocks_col = int(nbcols / nbcols_t)
        nb_blocks_zero = nb_blocks_col - len(toeplitz_matrices)
        nb_blocks_zero_left = 0

        all_zero_block_repartitions = list()
        while nb_blocks_zero >= 0:
            all_zero_block_repartitions.append((nb_blocks_zero_left, nb_blocks_zero))
            nb_blocks_zero -= 1
            nb_blocks_zero_left += 1

        my_central_row = np.concatenate(
            [toeplitz_matrix for toeplitz_matrix in toeplitz_matrices],
            axis=1)

        m = np.bmat([
            [zero_toeplitz for _ in range(rep[0])] + [my_central_row] + [zero_toeplitz for _ in range(rep[1])]
            for rep in all_zero_block_repartitions
        ])

        print(m)


class SoftMaxLayer(Layer):
    def __init__(self):
        super().__init__(
            func=nn.Softmax(dim=1),
            graph_layer=False
        )

    def get_matrix(self):
        raise NotImplementedError()

#################
# Architectures #
#################


class Architecture(nn.Module):

    def __init__(self,
                 layers: List[Layer],
                 preprocess: Callable):
        super().__init__()
        self.layers = layers
        self.preprocess = preprocess

        for i, layer in enumerate(layers):
            for j, param in enumerate(layer.func.parameters()):
                self.register_parameter(f"{i}_{j}", param)

    def forward(self, x, store_for_graph=False):
        # List to store intermediate results if needed
        x = self.preprocess(x)
        # Going through all layers
        for layer in self.layers:
            x = layer.process(x.double(), store_for_graph=store_for_graph)
        # Returning final result
        return x

    def get_graph_values(self, x):
        # Processing sample
        self.forward(x, store_for_graph=True)
        # Getting matrix for each layer
        ret = dict()
        i = 0
        for layer in self.layers:
            if layer.graph_layer:
                m = layer.get_matrix()
                ret[i] = m
                i += 1
        return ret


def mnist_preprocess(x):
    return x.view(-1, 28 * 28)


mnist_mlp = Architecture(
        preprocess=mnist_preprocess,
        layers=[
            LinearLayer(28*28, 500),
            LinearLayer(500, 256),
            LinearLayer(256, 10),
            SoftMaxLayer()
])
