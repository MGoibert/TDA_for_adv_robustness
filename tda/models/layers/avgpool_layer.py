from .layer import Layer
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix


class AvgPool2dLayer(Layer):
    def __init__(self, kernel_size, stride=None, activ=None):

        self._activ = activ
        self._k = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )

        if stride is None:
            self._stride = self._k
        else:
            self._stride = stride if isinstance(stride, tuple) else (stride, stride)

        super().__init__(
            func=nn.AvgPool2d(kernel_size=kernel_size, stride=stride), graph_layer=True
        )

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for AvgPool2dLayers
        # (it depends dynamically on the input since it's an avg ope)
        self.matrix = None

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        dim = 1
        for d in self._activations_shape:
            dim *= d

        nb_step_x = (self._activations_shape[-1] - self._k[1]) // self._stride[1]
        nb_step_y = self._activations_shape[-2] - self._k[0] // self._stride[0]

        dim_out = nb_step_x * nb_step_y

        m = np.zeros((dim, dim_out))
        for idx_out in range(dim_out):
            nx = idx_out % nb_step_x
            ny = idx_out // nb_step_x
            idx = [
                nx * self._stride[1]
                + idx_x
                + self._activations_shape[-1] * (ny * self._stride[0] + idx_y)
                for idx_y in range(self._k[0])
                for idx_x in range(self._k[1])
            ]
            m[:, idx_out][idx] = (
                1.0
                / (self._k[0] * self._k[1])
                * self._activation_values.flatten().cpu().detach()[idx]
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
