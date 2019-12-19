#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:24:22 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""


import logging
from functools import reduce
from typing import List, Callable, Tuple, Dict

import numpy as np
from cached_property import cached_property
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import toeplitz

torch.set_default_tensor_type(torch.DoubleTensor)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


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
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))


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
        return {
            parentidx: np.abs((self._activations[parentidx] * m).detach().numpy())
            for parentidx in self._activations
        }

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        _x = {key: x[key].reshape(-1, self._in_width) for key in x}
        if store_for_graph:
            self._activations = _x
        _x = sum(_x.values())
        if self._activ:
            out = self.func(_x)
            if type(a) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
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
        return {
            parentidx: np.matrix(m.transpose())
            for parentidx in self._parent_indices
        }

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        parent_indices = list(x.keys())
        x = sum(x.values()).double()
        out, indx = self.func(x)
        if store_for_graph:
            self._parent_indices = parent_indices
            self._activations_shape = x.shape
            self._indx = indx
            self._out_shape = out.shape
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out

class AvgPool2dLayer(Layer):

    def __init__(self, kernel_size, activ=None):

        self._activ = activ
        self._k = kernel_size

        super().__init__(
            func=nn.AvgPool2d(kernel_size=kernel_size),
            graph_layer=True
        )

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        dim_out = 1
        for d in self._out_shape:
            dim_out *= d
        m = np.zeros((dim, dim_out))
        for idx_out in range(dim_out):
            idx = [self._k*idx_out+i+j*self._activations_shape[-1] for j in range(self._k) for i in range(self._k)]
            m[:, idx_out][idx] = 1.0/(2*self._k) * self._activation_values.flatten()[idx]
        return {
            parentidx: np.matrix(m.transpose())
            for parentidx in self._parent_indices
        }

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        parent_indices = list(x.keys())
        x = sum(x.values()).double()
        out = self.func(x)
        if store_for_graph:
            self._parent_indices = parent_indices
            self._activation_values = x
            self._activations_shape = x.shape
            self._out_shape = out.shape
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
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

        m = {
            parentidx: np.bmat(tuple(
            tuple(self.get_matrix_for_channel(in_c, out_c)[parentidx] for in_c in range(self._in_channels))
            for out_c in range(self._out_channels)
        ))
            for parentidx in self._activations
        }

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

        # all shapes should be the same in the activations
        # let's choose one parent layer
        parentidx = list(self._activations.keys())[0]

        activations = self._activations[parentidx][:, in_channel, :, :]
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

        return {
            parentidx: np.abs(self._activations[parentidx][:, in_channel, :, :].detach().numpy().reshape(-1) * np.array(m))
            for parentidx in self._activations
            }

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        x = sum(x.values())
        if self._activ:
            out = self.func(x)
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
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


class DropOut(Layer):
    def __init__(self):
        super().__init__(
            func=nn.Dropout(),
            graph_layer=False
        )

    def get_matrix(self):
        raise NotImplementedError()

class BatchNorm2d(Layer):
    def __init__(self, channels):
        super().__init__(
            func=nn.BatchNorm2d(num_features=channels),
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
                 layer_links: List[Tuple[int, int]] = None,
                 name: str = ""):
        """
        Instatiating architecture with a list of layers and edges.
        The graph on the layers should be a DAG (we won't check for cycles)
        """
        super().__init__()
        self.name = name
        self.layers = layers
        self.layer_links = layer_links or [(-1, 0)] + [(i, i + 1) for i in range(len(layers) - 1)]

        self.preprocess = preprocess

        self.layer_visit_order = Architecture.walk_through_dag(self.layer_links)
        self.parent_dict = Architecture.get_parent_dict(self.layer_links)

        for i, layer in enumerate(layers):
            for j, param in enumerate(layer.func.parameters()):
                self.register_parameter(f"{i}_{j}", param)

    def get_pre_softmax_idx(self):
        softmax_layer_idx = None
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, SoftMaxLayer):
                softmax_layer_idx = layer_idx
                break
        if softmax_layer_idx is None:
            raise RuntimeError(f"Didn't find any SoftMax in {self.layers}")

        return [link for link in self.layer_links if link[1] == softmax_layer_idx][0][0]


    @staticmethod
    def walk_through_dag(
        edges: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Helper function to build an ordered walkthrough in the DAG
        """

        all_nodes = set([edge[0] for edge in edges]).union(set([edge[1] for edge in edges]))

        # Step 1: find the roots and add them to the stack
        stack = [node for node in all_nodes if not any([node == edge[1] for edge in edges])]

        order = list()

        while len(stack) > 0:
            current_node = stack.pop()
            order.append(current_node)

            for child in [edge[1] for edge in edges if edge[0] == current_node]:
                all_parents = [edge[0] for edge in edges if edge[1] == child]
                if all([parent in order for parent in all_parents]):
                    stack.append(child)

        return order

    @staticmethod
    def get_parent_dict(
            edges: List[Tuple[int, int]]
    ):
        ret = dict()

        for node in [edge[1] for edge in edges]:
            parents = [edge[0] for edge in edges if edge[1] == node]
            ret[node] = sorted(parents)

        return ret

    def forward(self, x, store_for_graph=False, output="final"):
        # List to store intermediate results if needed
        x = self.preprocess(x)

        outputs = {-1: x.double()}

        # Going through all layers
        for layer_idx in self.layer_visit_order:
            if layer_idx != -1:
                layer = self.layers[layer_idx]
                input = {
                    parent_idx: outputs[parent_idx].double()
                    for parent_idx in self.parent_dict[layer_idx]
                }
                outputs[layer_idx] = layer.process(input, store_for_graph=store_for_graph)

        # Returning final result
        if output == "presoft":
            return outputs[self.layer_visit_order[-2]]
        elif output == "all_inner":
            return outputs
        elif output == "final":
            return outputs[self.layer_visit_order[-1]]
        else:
            raise RuntimeError(f"Unknown output type {output}")

    def get_graph_values(self, x) -> Dict:
        # Processing sample
        # logging.info(f"Shape of x is {x.shape}")
        self.forward(x, store_for_graph=True)
        # Getting matrix for each layer
        ret = dict()
        for layer_idx, layer in enumerate(self.layers):
            if layer.graph_layer:
                m = layer.get_matrix()
                for parentidx in m:
                    ret[(parentidx, layer_idx)] = m[parentidx]
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

def mnist_preprocess2(x):
    return x.view(-1, 1, 28, 28)

mnist_res = Architecture(
    name="mnist_res",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 500),
        LinearLayer(500, 600),
        LinearLayer(600, 10),
        SoftMaxLayer()
    ],
    layer_links=[
    (-1,0), (0,1), (1,2), (0,2), (2,3)
    ])


mnist_mlp = Architecture(
    name="simple_fcn_mnist",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 500),
        LinearLayer(500, 256),
        LinearLayer(256, 10),
        SoftMaxLayer()
    ])

mnist_small_mlp = Architecture(
    name="small_mlp",
    preprocess=mnist_preprocess,
    layers=[
        LinearLayer(28 * 28, 200),
        LinearLayer(200, 50),
        LinearLayer(50, 10),
        SoftMaxLayer()
    ])

mnist_lenet = Architecture(
    name="mnist_lenet",
    preprocess=mnist_preprocess2,
    layers=[
        ConvLayer(1, 10, 5, activ=F.relu),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(10, 20, 5, activ=F.relu),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        LinearLayer(320, 50, activ=F.relu),
        DropOut(),
        LinearLayer(50, 10),
        SoftMaxLayer()
    ])

mnist_test = Architecture(
    name="mnist_test",
    preprocess=mnist_preprocess2,
    layers=[
        ConvLayer(1, 10, 5, activ=F.relu),  # output 10 * 28 * 28
        BatchNorm2d(10),
        ConvLayer(10, 20, 5, activ=F.relu),  # output 6 * 28 * 28
        AvgPool2dLayer(2),
        LinearLayer(20*20*5, 10),
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
    mnist_res,
    mnist_test,
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet,
    mnist_lenet,
    mnist_small_mlp
]


def get_architecture(architecture_name: str) -> Architecture:
    for archi in known_architectures:
        if architecture_name == archi.name:
            return archi
