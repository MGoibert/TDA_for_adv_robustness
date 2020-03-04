#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numba import njit
from scipy.sparse import coo_matrix, bmat as sparse_bmat

from tda.tda_logging import get_logger

torch.set_default_tensor_type(torch.DoubleTensor)
logger = get_logger("Layers")


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
            ret[parentidx] = coo_matrix(np.abs(weight.cpu().detach().numpy()))

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

        self._activ = activ

        super().__init__(
            func=nn.MaxPool2d(kernel_size, return_indices=True),
            graph_layer=True
        )

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        idx = self._indx
        idx = idx.cpu().numpy().flatten()
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
    @njit
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