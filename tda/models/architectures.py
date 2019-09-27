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
from collections import OrderedDict

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
