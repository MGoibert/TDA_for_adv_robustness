from .layer import Layer
from torch import nn
from functools import reduce
from numba import njit
import numpy as np
from scipy.sparse import coo_matrix, bmat as sparse_bmat
from tda.tda_logging import get_logger
import torch
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)

logger = get_logger("ConvLayer")


class ConvLayer(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        input_shape=None,
        stride=1,
        padding=0,
        bias=False,
        activ=None,
        name=None,
        grouped_channels: bool = False,
        p=0.0,
    ):

        if grouped_channels is True:
            groups = in_channels
            if in_channels != out_channels:
                raise RuntimeError(
                    "We don't support groups with different number of chans in input/output"
                )
        else:
            groups = 1

        if p > 0.0:
            func = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                ),
                nn.Dropout(p=p),
            )
        else:
            func = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )

        super().__init__(
            func=func, graph_layer=True, name=name,
        )

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._activ = activ
        self._stride = stride
        self._padding = padding
        self._input_shape = input_shape
        self._grouped_channels = grouped_channels

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
        padding: int,
    ):
        # logger.info(f"In _generate_cnn_edges")

        nbrows_kernel = kernel.shape[-2]
        nbcols_kernel = kernel.shape[-1]

        nbcols_input_with_padding = nbcols_input + 2 * padding
        nbrows_input_with_padding = nbrows_input + 2 * padding
        nbcols_with_padding = nbrows_input_with_padding * nbcols_input_with_padding

        offsets_x = range(0, nbcols_input - nbcols_kernel + 1 + 2 * padding, stride)

        offsets_y = range(
            0, int(nbcols / nbcols_input) - nbrows_kernel + 1 + 2 * padding, stride
        )

        nb_offset_x = len(offsets_x)
        nb_offset_y = len(offsets_y)

        data = list()
        row_ind = list()
        col_ind = list()
        for i in range(nbrows_kernel):
            for j in range(nbcols_kernel):
                for offset_x in offsets_x:
                    for offset_y in offsets_y:
                        row = offset_x // stride + (offset_y // stride) * nb_offset_x
                        col = (
                            offset_x
                            + j
                            + offset_y * nbcols_input_with_padding
                            + i * nbcols_input_with_padding
                        )

                        if padding > 0:
                            if (
                                col < nbcols_input_with_padding * padding
                                or col % nbcols_input_with_padding
                                in list(range(padding))
                                + list(
                                    range(
                                        nbcols_input_with_padding - padding,
                                        nbcols_input_with_padding,
                                    )
                                )
                                or col
                                >= nbcols_with_padding
                                - nbcols_input_with_padding * padding
                            ):
                                continue
                            else:
                                col = (
                                    col
                                    - nbcols_input_with_padding * padding
                                    - (offset_y + i - padding) * 2 * padding
                                    - padding
                                )

                        col_ind.append(col)
                        row_ind.append(row)
                        data.append(kernel[i, j])

                        if (
                            not 0 <= row_ind[-1] <= nb_offset_x * nb_offset_y - 1
                            or not 0 <= col_ind[-1] <= nbcols - 1
                        ):
                            raise RuntimeError("Invalid edge")
        return data, col_ind, row_ind, nb_offset_x * nb_offset_y

    def build_matrix_for_channel(self, in_channel, out_channel):
        """
        Return the weight of unrolled weight matrix
        for a convolutional layer
        """

        ##############################################
        # Selecting in and out channel in the kernel #
        ##############################################

        # logging.info(f"Processing in={in_channel} and out={out_channel}")
        # logger.info(f"In build_matrix_for_channel")

        kernel = None
        grouped_channels = hasattr(self, "_grouped_channels") and self._grouped_channels

        for name, param in self.func.named_parameters():
            # logger.info(name)
            # logger.info(f"size param {param[1].size()} and name = {param[0]}")
            # logger.info(f"out channel = {out_channel} and in channel = {in_channel}")
            # TODO: why this order out / in ???
            if "weight" in name:
                if not grouped_channels:
                    kernel = param.data[out_channel, in_channel, :, :]
                else:
                    kernel = param.data[out_channel, 0, :, :]
                # logger.info(f"name = {name_} and kernel = {kernel.size()}")

        assert kernel is not None

        ##################################
        # Compute the size of the matrix #
        ##################################

        nbcols_input = self._input_shape[1]
        nbrows_input = self._input_shape[0]
        nbcols = self._input_shape[0] * self._input_shape[1]

        #############################
        # Compute Toeplitz matrices #
        #############################

        (data, col_ind, row_ind, nbrows) = ConvLayer._generate_cnn_edges(
            nbcols=nbcols,
            nbcols_input=nbcols_input,
            nbrows_input=nbrows_input,
            kernel=kernel.detach().cpu().numpy(),
            padding=self._padding,
            stride=self._stride,
        )

        grouped_channels = hasattr(self, "_grouped_channels") and self._grouped_channels

        if grouped_channels and in_channel != out_channel:
            return coo_matrix(([], ([], [])), shape=(nbrows, nbcols))
        else:
            mat = coo_matrix((data, (row_ind, col_ind)), shape=(nbrows, nbcols))
            return mat

    def build_matrix(self) -> coo_matrix:
        matrix_grid = [
            [
                self.build_matrix_for_channel(in_c, out_c)
                for in_c in range(self._in_channels)
            ]
            for out_c in range(self._out_channels)
        ]
        self.matrix = sparse_bmat(matrix_grid)
        return self.matrix

    def process(self, x, store_for_graph):
        # logger.info(f"In process")
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        x = sum(x.values())

        if not hasattr(self, "_input_shape") or self._input_shape is None:
            logger.info(f"{self} received input with shape {x.shape}")
            self._input_shape = (x.shape[-2], x.shape[-1])

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
