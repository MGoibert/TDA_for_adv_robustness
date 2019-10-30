#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:24:22 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""


from functools import reduce
from typing import List, Callable

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def __init__(self, in_width, out_width, activ=None):

        super().__init__(
            func=nn.Linear(in_width, out_width),
            graph_layer=True
        )

        self._in_width = in_width
        self._activ = activ

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        m = list(self.func.parameters())[0]
        return np.abs((self._activations * m).detach().numpy())

    def process(self, x, store_for_graph):
        _x = x.reshape(-1, self._in_width)
        if store_for_graph:
            self._activations = _x
        if self._activ:
            return self._activ(self.func(_x))
        else:
            return self.func(_x)


class MaxPool2dLayer(Layer):

    def __init__(self, kernel_size, activ=None):

        self._activ = activ

        super().__init__(
            func=nn.MaxPool2d(kernel_size, return_indices=True),
            graph_layer=True
        )

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        idx = self._indx.numpy().flatten()
        dim = 1
        dim_out = 1
        for d in self._activations_shape:
            dim *= d
        for d in self._out_shape:
            dim_out *= d
        # print("dim =", dim, "and dim output =", dim_out)
        m = np.zeros((dim, dim_out))
        for i in range(dim_out):
            m[:, i][idx[i]] = 1
        return np.matrix(m.transpose())

    def process(self, x, store_for_graph):
        out, indx = self.func(x)
        if store_for_graph:
            self._activations_shape = x.shape
            self._indx = indx
            self._out_shape = out.shape
        if self._activ:
            return self._activ(out)
        else:
            return out


class ConvLayer(Layer):

    def __init__(self, in_channels, out_channels, kernel_size, activ=None):

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

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activ = activ

    @staticmethod
    def _get_nb_elements(t):
        """
        Return the number of element of a given tensor
        """
        return reduce(lambda a, b: a * b, list(t.shape))

    def get_matrix(self):

        m = np.bmat(tuple(
            tuple(self.get_matrix_for_channel(in_c, out_c) for in_c in range(self._in_channels))
            for out_c in range(self._out_channels)
        ))

        return m

    def get_matrix_for_channel(self,
                               in_channel,
                               out_channel):
        """
        Return the weight of unrolled weight matrix
        for a convolutional layer
        """

        ##############################################
        # Selecting in and out channel in the kernel #
        ##############################################

        # logging.info(f"Processing in={in_channel} and out={out_channel}")

        for param in self.func.parameters():
            # TODO: why this order out / in ???
            kernel = param.data[out_channel, in_channel, :, :]
        # logging.info(f"My kernel for in={in_channel} and out={out_channel} has shape {kernel.shape}")

        ##################################
        # Compute the size of the matrix #
        ##################################

        activations = self._activations[:, in_channel, :, :]
        # logging.info(f"My activations for in={in_channel} and out={out_channel} has shape {activations.shape}")

        nbcols = ConvLayer._get_nb_elements(activations)
        # logging.info(f"NbCols={nbcols}")

        #############################
        # Compute Toeplitz matrices #
        #############################

        nbrows_t = list(activations.shape)[-1] - list(kernel.shape)[-1] + 1
        nbcols_t = list(activations.shape)[-1]
        # logging.info(f"The Toeplitz matrices are {nbrows_t}x{nbcols_t}")

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

        ##############################
        # Stacking Toeplitz matrices #
        ##############################

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

        m = np.bmat(tuple(
            tuple(zero_toeplitz for _ in range(rep[0])) + (my_central_row,) + tuple(zero_toeplitz for _ in range(rep[1]))
            for rep in all_zero_block_repartitions
        ))

        return np.abs(activations.detach().numpy().reshape(-1) * np.array(m))

    def process(self, x, store_for_graph):
        if store_for_graph:
            self._activations = x
        if self._activ:
            return self._activ(self.func(x))
        else:
            return self.func(x)


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
                 preprocess: Callable,
                 name: str = ""):
        super().__init__()
        self.name = name
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
        # logging.info(f"Shape of x is {x.shape}")
        self.forward(x, store_for_graph=True)
        # Getting matrix for each layer
        ret = list()
        for layer in self.layers:
            if layer.graph_layer:
                # logging.info(f"Processing layer {layer}")
                m = layer.get_matrix()
                ret.append(m)
        return ret

    def get_nb_graph_layers(self) -> int:
        return sum([
            1 for layer in self.layers if layer.graph_layer
        ])


#######################
# MNIST Architectures #
#######################

def mnist_preprocess(x):
    return x.view(-1, 28 * 28)


mnist_mlp = Architecture(
    name="simple_fcn_mnist",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 500),
        LinearLayer(500, 256),
        LinearLayer(256, 10),
        SoftMaxLayer()
    ])


######################
# SVHN Architectures #
######################


def svhn_preprocess(x):
    return x.reshape(-1, 3, 32, 32)


svhn_cnn_simple = Architecture(
    name="simple_cnn_svhn",
    preprocess=svhn_preprocess,
    layers=[
        ConvLayer(3, 8, 5),  # output 8 * 28 * 28
        ConvLayer(8, 3, 5),  # output 3 * 24 * 24
        LinearLayer(3 * 24 * 24, 500),
        LinearLayer(500, 256),
        LinearLayer(256, 10),
        SoftMaxLayer()
    ])

svhn_lenet = Architecture(
    name="svhn_lenet",
    preprocess=svhn_preprocess,
    layers=[
        ConvLayer(3, 6, 5, activ=F.relu),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(6, 16, 5, activ=F.relu),
        MaxPool2dLayer(2),  # output 16 * 5 * 5
        LinearLayer(16 * 5 * 5, 120, activ=F.relu),
        LinearLayer(120, 84, activ=F.relu),
        LinearLayer(84, 10),
        SoftMaxLayer()
    ])

known_architectures: List[Architecture] = [
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet
]


def get_architecture(architecture_name: str) -> Architecture:
    for archi in known_architectures:
        if architecture_name == archi.name:
            return archi
