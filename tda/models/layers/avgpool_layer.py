from .layer import Layer
from torch import nn
import numpy as np
from scipy.sparse import coo_matrix, bmat as sparse_bmat


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
            func=nn.AvgPool2d(kernel_size=kernel_size, stride=stride, ceil_mode=True), graph_layer=True
        )

    @staticmethod
    def build_matrix_for_channel(
        activations_shape, kernel_size, strides, in_channel, out_channel
    ) -> coo_matrix:
        """
                Return the weight of the linear layer, ignore biases
                """
        dim = activations_shape[2] * activations_shape[3]

        nb_step_x = 1 + (activations_shape[-1] - kernel_size[1]) // strides[1]
        nb_step_y = 1 + (activations_shape[-2] - kernel_size[0]) // strides[0]

        dim_out = nb_step_x * nb_step_y

        m = np.zeros((dim, dim_out))

        if in_channel == out_channel:
            for idx_out in range(dim_out):
                nx = idx_out % nb_step_x
                ny = idx_out // nb_step_x
                idx = [
                    nx * strides[1]
                    + idx_x
                    + activations_shape[-1] * (ny * strides[0] + idx_y)
                    for idx_y in range(kernel_size[0])
                    for idx_x in range(kernel_size[1])
                ]
                m[:, idx_out][idx] = 1.0 / (kernel_size[0] * kernel_size[1])

        return coo_matrix(np.matrix(m.transpose()))

    def build_matrix(self) -> coo_matrix:
        nb_channels = self._activations_shape[1]
        matrix_grid = [
            [
                AvgPool2dLayer.build_matrix_for_channel(
                    activations_shape=self._activations_shape,
                    kernel_size=self._k,
                    strides=self._stride,
                    in_channel=in_c,
                    out_channel=out_c,
                )
                for in_c in range(nb_channels)
            ]
            for out_c in range(nb_channels)
        ]
        self.matrix = sparse_bmat(matrix_grid)
        return self.matrix

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        x_sum = sum(x.values()).double()
        out = self.func(x_sum)
        if store_for_graph:
            self._activations = x
            self._activations_shape = x_sum.shape
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out
