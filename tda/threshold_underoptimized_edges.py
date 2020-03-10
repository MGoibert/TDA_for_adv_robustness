import typing

import numpy as np
import torch
from scipy.sparse import coo_matrix

from tda.models import Architecture
from tda.tda_logging import get_logger
from tda.embeddings import ThresholdStrategy

logger = get_logger("Thresholds Underoptimized")


def _process_raw_quantiles(
    raw_quantiles: str, architecture: Architecture = None
) -> typing.Dict[int, float]:

    if not "_" in raw_quantiles:
        # Uniform quantile
        return {
            layer_idx: round(float(raw_quantiles), 2)
            for layer_idx in architecture.layer_visit_order
        }

    ret = dict()
    for raw_quantile in raw_quantiles.split("_"):
        layer_idx, value = raw_quantile.split(":")
        ret[int(layer_idx)] = float(value)
    return ret


def underopt_edges(
    quantiles: typing.Dict, method: str, model: Architecture, model_init: Architecture
):
    limit_val = dict()
    qtest = dict()
    underoptimized_edges = dict()
    for layer_idx, layer in enumerate(model.layers):
        if "weight" in layer.func.state_dict():
            param = layer.func.state_dict()["weight"]
            param_init = model_init.layers[layer_idx].func.state_dict()["weight"]

            if method == ThresholdStrategy.UnderoptimizedMagnitudeIncrease:
                limit_val[layer_idx] = torch.abs(param) - torch.abs(param_init)
            elif method == ThresholdStrategy.UnderoptimizedLargeFinal:
                limit_val[layer_idx] = torch.abs(param)
            qtest[layer_idx] = np.quantile(limit_val[layer_idx], quantiles[layer_idx])
            underoptimized_edges[layer_idx] = (
                (limit_val[layer_idx] < qtest[layer_idx])
                .nonzero()
                .cpu()
                .numpy()
                .tolist()
            )

    return underoptimized_edges


def kernel_to_edge_idx(kernel_idx, kernel_shape, mat_shape):
    # Kernel idx: [out_channel, in_channel, row, col]
    # Kernel shape: [nb_out_channel, nb_in_channel, nb_row, nb_col]
    # Matrix mat_shape: [output_total_size, input_total_size] ([5760, 784])
    size_col_block = mat_shape[1] / kernel_shape[1]  # corresponds to in channel
    size_row_block = mat_shape[0] / kernel_shape[0]  # corresponds to out channel

    edge_idx = list()
    for kidx in kernel_idx:
        # Ex: [1,0,0,4]
        rest_row = int(np.sqrt(mat_shape[1] / kernel_shape[1])) - kernel_shape[2]
        rest_col = int(np.sqrt(mat_shape[1] / kernel_shape[1])) - kernel_shape[3]
        nb_apply_row = rest_row + 1
        nb_apply_col = rest_col + 1
        first_apply = int(np.sqrt(mat_shape[1] / kernel_shape[1])) * kidx[2] + kidx[3]
        total_idx_col = list()
        for idx_col in range(nb_apply_col):
            begin = (
                idx_col * (int(np.sqrt(mat_shape[1] / kernel_shape[1]))) + first_apply
            )
            end = begin + nb_apply_row
            total_idx_col.append(list(range(begin, end)))
        total_idx_col = [
            int(size_col_block * kidx[1]) + item
            for sublist in total_idx_col
            for item in sublist
        ]
        total_idx_row = list(
            int(size_row_block * kidx[0])
            + np.asarray(range(nb_apply_row * nb_apply_col))
        )
        total_idx = list(map(list, zip(total_idx_row, total_idx_col)))
        edge_idx += total_idx
    return edge_idx


def process_thresholds_underopt(
    raw_thresholds: str, architecture: Architecture, method: str,
) -> typing.Dict:
    """

    :param method:
    :param raw_thresholds:
    :param architecture:
    :return:
    """

    architecture_init = architecture.get_initial_model()

    quantiles_per_layer = _process_raw_quantiles(raw_thresholds, architecture)
    underoptimized_edges = underopt_edges(
        quantiles=quantiles_per_layer,
        method=method,
        model=architecture,
        model_init=architecture_init,
    )

    if architecture.name in ["mnist_lenet", "fashion_mnist_lenet"]:
        mat_shapes = [[5760, 784], [1280, 1440]]
    elif architecture.name in ["svhn_lenet", "cifar_lenet"]:
        mat_shapes = [[4704, 3072], [1600, 1176]]
    else:
        raise NotImplementedError(f"This function cannot handle {architecture.name}")

    # Post-processing the ConvLayers
    c = 0
    for layer_idx, layer in enumerate(architecture.layers):
        if isinstance(layer.func, torch.nn.Conv2d):
            kernel_shape = layer.func.weight.size()
            underoptimized_edges[layer_idx] = kernel_to_edge_idx(
                underoptimized_edges[layer_idx], kernel_shape, mat_shapes[c]
            )
            c += 1

    logger.info(f"Keys = {underoptimized_edges.keys()}")
    logger.info(
        f"Size edges kept = {[len(underoptimized_edges[k]) for k in underoptimized_edges.keys()]}"
    )

    return underoptimized_edges


def thresholdize_underopt_v2(
    raw_thresholds: str, architecture: Architecture, method: str,
):
    """

    :param method:
    :param raw_thresholds:
    :param architecture:
    :return:
    """

    quantiles_per_layer = _process_raw_quantiles(raw_thresholds)

    architecture_init = architecture.get_initial_model()

    matrices = architecture.get_layer_matrices()
    matrices_init = architecture_init.get_layer_matrices()

    for layer_idx in matrices.keys():

        if matrices[layer_idx] is not None:

            arr = np.array(matrices[layer_idx].data)
            arr_init = np.array(matrices_init[layer_idx].data)

            if method == ThresholdStrategy.UnderoptimizedMagnitudeIncreaseV2:
                value = np.abs(arr) - np.abs(arr_init)
            elif method == ThresholdStrategy.UnderoptimizedLargeFinalV2:
                value = np.abs(arr)
            else:
                raise NotImplementedError(f"Unknown method {method}")

            min_value = np.quantile(value, quantiles_per_layer[layer_idx])

            loc = arr < min_value

            # Thresholding the matrix
            logger.info(f"Applying underopt threshold to layer {layer_idx} !")
            matrix_shape = architecture.layers[layer_idx].matrix.shape
            logger.info(f"Matrix shape is {matrix_shape}")

            architecture.layers[layer_idx].matrix = coo_matrix(
                (
                    architecture.layers[layer_idx].matrix.data[loc],
                    (
                        architecture.layers[layer_idx].matrix.row[loc],
                        architecture.layers[layer_idx].matrix.col[loc],
                    ),
                ),
                shape=matrix_shape,
            )
