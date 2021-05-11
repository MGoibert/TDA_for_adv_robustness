from collections import defaultdict

from scipy.sparse import coo_matrix, bmat as sparse_bmat
from torch import nn

from .layer import Layer
import torch
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)


class MaxPool2dLayer(Layer):
    def __init__(self, kernel_size, stride=None, activ=None):

        if stride is None:
            stride = kernel_size

        super().__init__(
            func=nn.MaxPool2d(kernel_size, stride, return_indices=True),
            graph_layer=True,
        )

        self._activ = activ
        # self._use_activation = True

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for MaxPool2dLayers
        self.matrix = None

    def get_matrix_for_channel(self, in_c, out_c):
        """
        Return the weight of the linear layer, ignore biases
        """
        dim = 1
        dim_out = 1
        for d in self._activations_shape[2:]:
            dim *= d
        for d in self._out_shape[2:]:
            dim_out *= d

        if in_c != out_c:
            return {
                parentidx: coo_matrix(([], ([], [])), shape=(dim_out, dim))
                for parentidx in self._parent_indices
            }

        # We should build matrices point per point
        assert self._activations_shape[0] == 1

        matrices = dict()

        for parentidx in self._parent_indices:

            data = list()
            cols = list()
            rows = list()

            for i in range(dim_out):
                rows.append(i)
                idx = self._indx[0, in_c, :, :].reshape(-1).cpu().detach().numpy()[i]
                cols.append(idx)

                row_in_source = idx // self._activations_shape[3]
                col_in_source = idx % self._activations_shape[3]
                data.append(
                    self._activations[parentidx]
                    .cpu()
                    .detach()
                    .numpy()[0, in_c, row_in_source, col_in_source]
                )

            matrices[parentidx] = coo_matrix((data, (rows, cols)), shape=(dim_out, dim))

        return matrices

    def get_matrix(self):

        all_matrices = defaultdict(lambda: dict())
        for in_c in range(self._activations_shape[1]):
            for out_c in range(self._out_shape[1]):
                matrice_dict_for_channels = self.get_matrix_for_channel(in_c, out_c)
                for parent_idx in matrice_dict_for_channels:
                    all_matrices[parent_idx][(in_c, out_c)] = matrice_dict_for_channels[
                        parent_idx
                    ]

        matrice_dict = dict()

        for parent_idx in all_matrices:
            matrix_grid = [
                [
                    all_matrices[parent_idx][(in_c, out_c)]
                    for in_c in range(self._activations_shape[1])
                ]
                for out_c in range(self._out_shape[1])
            ]
            matrice_dict[parent_idx] = sparse_bmat(matrix_grid)
        return matrice_dict

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        x_sum = sum(x.values())
        out, indx = self.func(x_sum)
        if store_for_graph:
            self._activations = x
            self._indx = indx
            self._out_shape = out.shape
            self._activations_shape = x_sum.shape
            self._activations_values = x_sum
            self._parent_indices = list(x.keys())
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out
