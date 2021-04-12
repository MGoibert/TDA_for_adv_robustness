from . import AvgPool2dLayer
from .layer import Layer
from torch import nn
from scipy.sparse import coo_matrix


class AdaptativeAvgPool2dLayer(Layer):
    def __init__(self, output_size):

        self._output_size = output_size

        super().__init__(
            func=nn.AdaptiveAvgPool2d(output_size=self._output_size), graph_layer=True
        )

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for AvgPool2dLayers
        # (it depends dynamically on the input since it's an avg ope)
        self.matrix = None

    def get_matrix(self):
        return AvgPool2dLayer.get_matrix_from_values(
            activation_values=self._activation_values,
            activations_shape=self._activations_shape,
            kernel_size=self._k,
            strides=self._stride,
            parent_indices=self._parent_indices,
        )

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

            output_height, output_width = self._output_size
            input_height = self._activations_shape[0]
            input_width = self._activations_shape[1]

            stride_width = input_width // output_width
            kernel_width = input_width - (output_width - 1) * stride_width

            stride_height = input_height // output_height
            kernel_height = input_height - (output_height - 1) * stride_height

            self._stride = (stride_width, stride_height)
            self._k = (kernel_width, kernel_height)

        return out
