#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:24:22 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import logging
from functools import reduce
from typing import List, Callable, Tuple, Dict, Optional

import numpy as np
from scipy.sparse import coo_matrix, bmat as sparse_bmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit

from tda.devices import device

torch.set_default_tensor_type(torch.DoubleTensor)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Architecture")


##########
# Layers #
##########


class Layer(object):

    def __init__(self,
                 func: nn.Module,
                 graph_layer: bool,
                 name: Optional[str] = None
                 ):
        self.func = func
        self.graph_layer = graph_layer
        self._activations = None
        self.name = name

    def get_matrix(self):
        raise NotImplementedError()

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))


class LinearLayer(Layer):

    def __init__(self, in_width, out_width, activ=None, name=None):

        super().__init__(
            func=nn.Linear(in_width, out_width),
            graph_layer=True,
            name=name
        )

        self._in_width = in_width
        self._activ = activ

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        m = list(self.func.parameters())[0]

        ret = dict()

        for parentidx in self._activations:
            weight = self._activations[parentidx] * m
            if weight.is_cuda:
                weight = weight.cpu()
            ret[parentidx] = coo_matrix(np.abs(weight.detach().numpy()))

        return ret

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        _x = {key: x[key].reshape(-1, self._in_width) for key in x}
        if store_for_graph:
            self._activations = _x
        _x = sum(_x.values())
        if self._activ:
            out = self.func(_x)
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return self.func(_x)


class MaxPool2dLayer(Layer):

    def __init__(self, kernel_size, activ=None):

        super().__init__(
            func=nn.MaxPool2d(kernel_size, return_indices=True),
            graph_layer=True
        )

        self._activ = activ
        #self._use_activation = True

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        idx = self._indx
        for i in range(idx.shape[1]):
            idx[:,i,:,:] = idx[:,i,:,:]+i*self._activations_shape[2]*self._activations_shape[3]
        if idx.is_cuda:
            idx = idx.cpu()
        idx = idx.numpy().flatten()
        dim = 1
        dim_out = 1
        for d in self._activations_shape:
            dim *= d
        for d in self._out_shape:
            dim_out *= d
        # print("dim =", dim, "and dim output =", dim_out)
        m = np.zeros((dim, dim_out))
        for i in range(dim_out):
            if True: #self._use_activation:
                m[:, i][idx[i]] = self._out.flatten(0)[i] #self._activations.flatten(0)[idx[i]]
            else:
                m[:, i][idx[i]] = 1
        return {
            parentidx: coo_matrix(np.matrix(m.transpose()))
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
            self._activations = x
            self._out = out
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
        dim = 1
        for d in self._activations_shape:
            dim *= d
        for d in self._out_shape:
            dim_out *= d
        m = np.zeros((dim, dim_out))
        for idx_out in range(dim_out):
            idx = [self._k * idx_out + i + j * self._activations_shape[-1] for j in range(self._k) for i in
                   range(self._k)]
            m[:, idx_out][idx] = 1.0 / (2 * self._k) * self._activation_values.flatten()[idx]
        return {
            parentidx: coo_matrix(np.matrix(m.transpose()))
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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=False,
                 activ=None,
                 name=None):

        super().__init__(
            func=nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            graph_layer=True,
            name=name
        )

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activ = activ
        self._stride = stride
        self._padding = padding

    @staticmethod
    def _get_nb_elements(t):
        """
        Return the number of element of a given tensor
        """
        return reduce(lambda a, b: a * b, list(t.shape))

    @staticmethod
    @njit(parallel=True)
    def _generate_cnn_edges(
            kernel: np.ndarray,
            nbcols_input: int,
            nbrows_input: int,
            nbcols: int,
            stride: int,
            padding: int

    ):

        nbrows_kernel = kernel.shape[-2]
        nbcols_kernel = kernel.shape[-1]

        nbcols_input_with_padding = nbcols_input + 2 * padding
        nbrows_input_with_padding = nbrows_input + 2 * padding
        nbcols_with_padding = nbrows_input_with_padding * nbcols_input_with_padding

        offsets_i = range(
            0,
            nbcols_input - nbrows_kernel + 1 + 2 * padding,
            stride)

        offsets_j = range(
            0,
            int(nbcols / nbcols_input) - nbcols_kernel + 1 + 2 * padding,
            stride
        )

        nb_offset_i = len(offsets_i)
        nb_offset_j = len(offsets_j)

        data = list()
        row_ind = list()
        col_ind = list()
        for i in range(nbrows_kernel):
            for j in range(nbcols_kernel):
                for offset_i in offsets_i:
                    for offset_j in offsets_j:
                        row = offset_i // stride + (offset_j // stride) * nb_offset_i
                        col = offset_i + j + offset_j * (
                                    nb_offset_i + nbrows_kernel - 1) + i * nbcols_input_with_padding

                        if padding > 0:
                            if col < nbcols_input_with_padding \
                                    or col % nbcols_input_with_padding in (0, nbcols_input_with_padding - 1) \
                                    or col >= nbcols_with_padding - nbcols_input_with_padding:
                                continue
                            else:
                                col = col \
                                      - nbcols_input_with_padding \
                                      - padding \
                                      - (col // nbcols_input_with_padding - 1) * 2 * padding
                        col_ind.append(col)
                        row_ind.append(row)
                        data.append(kernel[i, j])

                        if not 0 <= row_ind[-1] <= nb_offset_i * nb_offset_j - 1 \
                                or not 0 <= col_ind[-1] <= nbcols - 1:
                            raise RuntimeError("Invalid edge")
        return data, col_ind, row_ind, nb_offset_i * nb_offset_j

    def get_matrix(self):

        m = dict()

        for parentidx in self._activations:
            matrix_grid = [
                [self.get_matrix_for_channel(in_c, out_c)[parentidx] for in_c in range(self._in_channels)]
                for out_c in range(self._out_channels)
            ]

            m[parentidx] = sparse_bmat(blocks=matrix_grid, format="coo")
            #logger.info(f"parent = {parentidx} and m = {m[parentidx].todense()}")

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

        for param_ in self.func.named_parameters():
            #logger.info(f"size param {param[1].size()} and name = {param[0]}")
            #logger.info(f"out channel = {out_channel} and in channel = {in_channel}")
            # TODO: why this order out / in ???
            param = param_[1]
            name_ = param_[0]
            if len(param.size()) > 1:
                kernel = param.data[out_channel, in_channel, :, :]
                #logger.info(f"name = {name_} and kernel = {kernel.size()}")

        ##################################
        # Compute the size of the matrix #
        ##################################

        parentidx = list(self._activations.keys())[0]
        activations = self._activations[parentidx][:, in_channel, :, :]

        nbcols_input = list(activations.shape)[-1]
        nbrows_input = list(activations.shape)[-2]
        nbcols = ConvLayer._get_nb_elements(activations)

        #############################
        # Compute Toeplitz matrices #
        #############################

        (data, col_ind, row_ind, nbrows) = ConvLayer._generate_cnn_edges(
            nbcols=nbcols,
            nbcols_input=nbcols_input,
            nbrows_input=nbrows_input,
            kernel=kernel.detach().cpu().numpy(),
            padding=self._padding,
            stride=self._stride
        )

        ret = dict()
        for parentidx in self._activations:
            activ = self._activations[parentidx][:, in_channel, :, :].reshape(-1)

            data_for_parent = [
                data[i] * float(activ[col_idx])
                for i, col_idx in enumerate(col_ind)
            ]

            ret[parentidx] = coo_matrix(
                (data_for_parent, (row_ind, col_ind)), shape=(nbrows, nbcols)
            )

        return ret

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


class ReluLayer(Layer):
    def __init__(self):
        super().__init__(
            func=nn.ReLU(),
            graph_layer=False
        )

    def process(self, x, store_for_graph):
        x = sum(x.values())
        out = self.func(x)
        return out

    def get_matrix(self):
        raise NotImplementedError()


class BatchNorm2d(Layer):
    def __init__(self, channels, activ=None):
        self._activ = activ
        super().__init__(
            func=nn.BatchNorm2d(num_features=channels),
            graph_layer=False
        )
        self._activ = activ

    def process(self, x, store_for_graph):
        x = sum(x.values())
        out = self.func(x)
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out

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
            layer_params = dict(layer.func.named_parameters())
            layer_name = layer.name or f"layer{i}"
            for name in layer_params:
                self.register_parameter(f"{layer_name}_{name}", layer_params[name])

    def set_train_mode(self):
        for layer in self.layers:
            layer.func.train()

    def set_eval_mode(self):
        for layer in self.layers:
            layer.func.eval()

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
        if device.type == "cuda":
            x = x.to(device)
        x = self.preprocess(x)

        outputs = {-1: x.double()}

        # Going through all layers
        for layer_idx in self.layer_visit_order:
            #logger.info(f"Layer nb {layer_idx}")
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
            # (f"Processing layer {layer_idx}")
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
        ConvLayer(1, 10, 5, activ=F.relu, bias=True, name="conv1"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        ConvLayer(10, 20, 5, activ=F.relu, bias=True, name="conv2"),  # output 6 * 28 * 28
        MaxPool2dLayer(2),
        LinearLayer(320, 50, activ=F.relu, name="fc1"),
        DropOut(),
        LinearLayer(50, 10, name="fc2"),
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

svhn_resnet = Architecture(
    name="svhn_resnet",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),

        #  Stack 1
        # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),

        # Stack 2
        # Block a
        ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128, activ=F.relu),
        ConvLayer(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=128),
        ReluLayer(),

        # Stack 3
        # Block a
        ConvLayer(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256, activ=F.relu),
        ConvLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=256),
        ReluLayer(),

        # Stack 4
        # Block a
        ConvLayer(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512),
        ReluLayer(),
        # Block b
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512, activ=F.relu),
        ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=512),
        ReluLayer(),

        # End part
        AvgPool2dLayer(kernel_size=4),
        LinearLayer(512, 10),
        SoftMaxLayer(),

        # Layer to reduce dimension in residual blocks
        ConvLayer(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False),
        ConvLayer(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False),
        ConvLayer(in_channels=256, out_channels=512, kernel_size=1, stride=2, padding=0, bias=False)
    ],
    layer_links=[(i-1,i) for i in range(45)]+[
        (1,6), (6,11), (16,21), (26,31), (36,41),
        (11,45), (45,16), (21,46), (46,26), (31,47), (47,36)
    ])

svhn_resnet_test = Architecture(
    name="svhn_resnet_test",
    preprocess=svhn_preprocess,
    layers=[
        # 1st layer / no stack or block
        ConvLayer(in_channels=3, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),

        #  Stack 1
            # Block a
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64, activ=F.relu),
        ConvLayer(in_channels=64, out_channels=64,kernel_size=3, stride=1, padding=1, bias=False),
        BatchNorm2d(channels=64),
        ReluLayer(),

        # End part
        AvgPool2dLayer(kernel_size=32),
        LinearLayer(64,10),
        SoftMaxLayer(),

        ],
    layer_links=[(i-1,i) for i in range(10)]+[
        (1,6)
    ])

known_architectures: List[Architecture] = [
    mnist_mlp,
    svhn_cnn_simple,
    svhn_lenet,
    svhn_resnet,
    mnist_lenet,
    mnist_small_mlp,
    svhn_resnet_test
]


def get_architecture(architecture_name: str) -> Architecture:
    for archi in known_architectures:
        if architecture_name == archi.name:
            return archi
