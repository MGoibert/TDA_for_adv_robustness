from . import AvgPool2dLayer
from .layer import Layer
from torch import nn
from scipy.sparse import coo_matrix, bmat as sparse_bmat


class AdaptativeAvgPool2dLayer(Layer):
    def __init__(self, output_size):

        self._output_size = output_size

        super().__init__(
            func=nn.AdaptiveAvgPool2d(output_size=self._output_size), graph_layer=True
        )

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

            output_height, output_width = self._output_size
            input_height = self._activations_shape[-2]
            input_width = self._activations_shape[-1]

            stride_width = input_width // output_width
            kernel_width = input_width - (output_width - 1) * stride_width

            stride_height = input_height // output_height
            kernel_height = input_height - (output_height - 1) * stride_height

            self._stride = (stride_width, stride_height)
            self._k = (kernel_width, kernel_height)

        return out
