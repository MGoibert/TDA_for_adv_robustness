from .layer import Layer
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix


class AvgPool2dLayer(Layer):
    def __init__(self, kernel_size, activ=None):

        self._activ = activ
        self._k = kernel_size

        super().__init__(func=nn.AvgPool2d(kernel_size=kernel_size), graph_layer=True)

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for AvgPool2dLayers
        self.matrix = None

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
            idx = [
                self._k * idx_out + i + j * self._activations_shape[-1]
                for j in range(self._k)
                for i in range(self._k)
            ]
            m[:, idx_out][idx] = (
                1.0 / (2 * self._k) * self._activation_values.flatten().cpu().detach()[idx]
            )
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
