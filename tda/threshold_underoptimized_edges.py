import os
import time
from typing import Dict, Tuple
from functools import reduce

import numpy as np
import torch
import mlflow
from numpy.random import Generator, PCG64

from tda.embeddings import ThresholdStrategy
from tda.models import Architecture
from tda.models.layers import ConvLayer
from tda.tda_logging import get_logger

logger = get_logger("Thresholds Underoptimized")


def _process_raw_quantiles(raw_quantiles: str) -> Dict[int, Tuple]:

    ret = dict()
    for raw_quantile in raw_quantiles.split("_"):
        layer_idx, value_low, value_up = raw_quantile.split(":")
        ret[int(layer_idx)] = (float(value_low), float(value_up))
        mlflow.log_metric(f"edges_quant_low_{int(layer_idx)}", float(value_low))
        mlflow.log_metric(f"edges_quant_up_{int(layer_idx)}", float(value_up))
    return ret


def underopt_edges(
    quantiles: Dict, method: str, model: Architecture, model_init: Architecture
):
    limit_val: Dict[int, torch.Tensor] = dict()
    underoptimized_edges = dict()
    for layer_idx, layer in enumerate(model.layers):

        weight_keys = [key for key in layer.func.state_dict() if "weight" in key]

        if len(weight_keys) > 0:
            param = layer.func.state_dict()[weight_keys[0]]
            param_init = model_init.layers[layer_idx].func.state_dict()[weight_keys[0]]

            if method == ThresholdStrategy.UnderoptimizedMagnitudeIncrease:
                limit_val[layer_idx] = torch.abs(param) - torch.abs(param_init)
            elif method == ThresholdStrategy.UnderoptimizedLargeFinal:
                limit_val[layer_idx] = torch.abs(param)
            elif method == ThresholdStrategy.UnderoptimizedRandom:
                n = reduce(lambda x, y: x * y, param.shape, 1)
                # Ensuring we select different edges each time
                gen = Generator(PCG64(int(time.time() + os.getpid())))
                limit_val[layer_idx] = (
                    torch.abs(param)
                    .reshape(-1)[gen.permutation(n)]
                    .reshape(param.shape)
                )
            limit_val[layer_idx] = limit_val[layer_idx].cpu()

            if layer_idx not in quantiles:
                underoptimized_edges[layer_idx] = list()
            else:
                low_quantile, up_quantile = quantiles[layer_idx]
                logger.info(f"[{layer_idx}] Quantiles are {low_quantile}-{up_quantile}")

                if low_quantile <= 0.0:
                    lower_bound = -np.infty
                else:
                    lower_bound = np.quantile(limit_val[layer_idx], low_quantile)

                if up_quantile >= 1.0:
                    upper_bound = np.infty
                else:
                    upper_bound = np.quantile(limit_val[layer_idx], up_quantile)

                underoptimized_edges[layer_idx] = (
                    torch.logical_and(
                        limit_val[layer_idx] < upper_bound,
                        limit_val[layer_idx] >= lower_bound,
                    )
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
    raw_thresholds: str, architecture: Architecture, method: str
) -> Dict:
    """

    :param quantile_number:
    :param thresholds_are_low_pass: if True keep underopt ; if False keep the opt edges
    :param method:
    :param raw_thresholds:
    :param architecture:
    :return:
    """

    architecture_init = architecture.get_initial_model()

    quantiles_per_layer = _process_raw_quantiles(raw_thresholds)
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
    elif architecture.name in ["svhn_lenet_bandw"]:
        mat_shapes = [[4704, 1024], [1600, 1176]]
    elif architecture.name in ["svhn_lenet_bandw2"]:
        mat_shapes = [[2352, 1024], [600, 588]]
    elif architecture.name in ["cifar_toy_resnet"]:
        mat_shapes = [[4704, 3072], [1600, 1176], [400, 4704]]
    elif architecture.name in ["cifar_resnet_1", "svhn_resnet_1"]:
        mat_shapes = [
            (65536, 3072),
            (65536, 65536),
            (65536, 65536),
            (65536, 65536),
            (65536, 65536),
            (32768, 32768),
            (32768, 32768),
            (32768, 32768),
            (32768, 32768),
            (16384, 16384),
            (16384, 16384),
            (16384, 16384),
            (16384, 16384),
            (8192, 8192),
            (8192, 8192),
            (8192, 8192),
            (8192, 8192),
            (32768, 65536),
            (16384, 32768),
            (8192, 16384),
        ]
    elif architecture.name in ["efficientnet"]:
        mat_shapes = [
            (32768, 12288),
            (32768, 32768),
            (16384, 32768),
            (16384, 16384),
            (16384, 16384),
            (98304, 16384),
            (24576, 98304),
            (6144, 24576),
            (36864, 6144),
            (36864, 36864),
            (6144, 36864),
            (36864, 6144),
            (36864, 36864),
            (6144, 36864),
            (36864, 6144),
            (9216, 36864),
            (2560, 9216),
            (15360, 2560),
            (15360, 15360),
            (2560, 15360),
            (15360, 2560),
            (15360, 15360),
            (15360, 15360),
            (15360, 15360),
            (3840, 15360),
            (1280, 3840),
            (7680, 1280),
            (7680, 7680),
            (1280, 7680),
            (7680, 1280),
            (7680, 7680),
            (1280, 7680),
            (7680, 1280),
            (7680, 7680),
            (1280, 7680),
            (7680, 1280),
            (7680, 7680),
            (1792, 7680),
            (10752, 1792),
            (10752, 10752),
            (1792, 10752),
            (10752, 1792),
            (10752, 10752),
            (1792, 10752),
            (10752, 1792),
            (10752, 10752),
            (1792, 10752),
            (10752, 1792),
            (2688, 10752),
            (768, 2688),
            (4608, 768),
            (4608, 4608),
            (768, 4608),
            (4608, 768),
            (4608, 4608),
            (768, 4608),
            (4608, 768),
            (4608, 4608),
            (768, 4608),
            (4608, 768),
            (4608, 4608),
            (768, 4608),
            (4608, 768),
            (4608, 4608),
            (1280, 4608),
            (7680, 1280),
            (7680, 7680),
            (1280, 7680),
            (5120, 1280),
            (6144, 16384),
            (2560, 6144),
            (1280, 15360),
            (1792, 1280),
            (768, 1792),
            (1280, 768),
        ]
    elif True:
        mat_shapes = [
            [16384, 3072],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [16384, 16384],
            [8192, 16384],
            [8192, 8192],
            [8192, 16384],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [8192, 8192],
            [4096, 8192],
            [4096, 4096],
            [4096, 8192],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
            [4096, 4096],
        ]
    else:
        mat_shapes = None

    # Post-processing the ConvLayers
    c = 0
    for layer_idx, layer in enumerate(architecture.layers):
        if isinstance(layer, ConvLayer):
            if isinstance(layer.func, torch.nn.Conv2d):
                kernel_shape = layer.func.weight.size()
            else:
                # Sequential case
                kernel_shape = layer.func[0].weight.size()

            mlflow.log_metric(
                f"kernel_nb_elems_{layer_idx}",
                reduce(lambda x, y: x * y, list(kernel_shape), 1),
            )

            underoptimized_edges[layer_idx] = kernel_to_edge_idx(
                underoptimized_edges[layer_idx], kernel_shape, mat_shapes[c]
            )
            c += 1

    logger.info(f"Keys = {underoptimized_edges.keys()}")
    logger.info(
        f"Size edges kept = {[len(underoptimized_edges[k]) for k in underoptimized_edges.keys()]}"
    )

    for k in underoptimized_edges:
        mlflow.log_metric(f"edges_kept_{k}", len(underoptimized_edges[k]))

    underoptimized_edges = {
        k: set([tuple(edge) for edge in underoptimized_edges[k]])
        for k in underoptimized_edges
    }

    return underoptimized_edges
