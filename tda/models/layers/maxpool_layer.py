from torch import nn
from .layer import Layer
import numpy as np
from scipy.sparse import coo_matrix


class MaxPool2dLayer(Layer):
    def __init__(self, kernel_size, activ=None):

        super().__init__(
            func=nn.MaxPool2d(kernel_size, return_indices=True), graph_layer=True
        )

        self._activ = activ
        # self._use_activation = True

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for MaxPool2dLayers
        self.matrix = None

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        idx = self._indx
        for i in range(idx.shape[1]):
            idx[:, i, :, :] = (
                idx[:, i, :, :]
                + i * self._activations_shape[2] * self._activations_shape[3]
            )
        idx = idx.cpu().numpy().flatten()
        dim = 1
        dim_out = 1
        for d in self._activations_shape:
            dim *= d
        for d in self._out_shape:
            dim_out *= d
        row = []
        col = []
        data = []
        for i in range(dim_out):
            col.append(i)
            row.append(idx[i])
            if True:  # self._use_activation:
                data.append(self._out.flatten(0)[i])
            else:
                data.append(1)
        m = coo_matrix((data, (row, col)), shape=(dim, dim_out),
                       dtype=np.float)
        m = m.T
        return {
            parentidx: m
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
