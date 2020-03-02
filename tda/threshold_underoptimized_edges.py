import typing
import numpy as np
import torch

from tda.graph_stats import get_stats
from tda.models import Architecture, Dataset
from tda.tda_logging import get_logger
from tda.rootpath import rootpath


logger = get_logger("Thresholds Underoptimized")


def process_thresholds_underopt(
        raw_thresholds: str,
        dataset: Dataset,
        architecture: Architecture,
        method: str = "magnitude_increase"
) -> typing.Dict:
    """
    Compute the actual thresholds to be used from a raw string like

    10_20_13

    OR

    0.1_10000_0.9

    It is assumed that if a threshold is between 0 and 1, it's a QUANTILE
    threshold (which required some stats to be computed on a sample of the
    dataset)

    :param dataset_size:
    :param raw_thresholds:
    :param dataset:
    :param architecture:
    :param epochs:
    :return:
    """
    def process(x):
        if x == "inf":
            return np.inf
        return float(x)

    model = architecture
    model_init = torch.load(f"{rootpath}/trained_models/{str(architecture)}_initial.model")

    def underopt_edges(q, method=method):
        limit_val = dict()
        qtest = dict()
        underoptimized_edges = dict()
        if isinstance(q, float):
            q = np.repeat(q, len([lk for lk in model.state_dict().keys() if 'weight' in lk]))
        i = 0
        for layer in model.state_dict().keys():
            if 'weight' in layer:
                if method == "magnitude_increase":
                    limit_val[layer] = torch.abs(model.state_dict()[layer]) - torch.abs(model_init[layer])
                elif method == "large_final":
                    limit_val[layer] = torch.abs(model.state_dict()[layer])
                qtest[layer] = np.quantile(limit_val[layer], q[i])
                underoptimized_edges[layer] = (limit_val[layer] < qtest[layer]).nonzero().numpy().tolist()
                i += 1
            
        return underoptimized_edges

    def kernel_to_edge_idx(kernel_idx, kernel_shape, mat_shape):
        # Kernel idx: [out_channel, in_channel, row, col]
        # Kernel shape: [nb_out_channel, nb_in_channel, nb_row, nb_col]
        # Matrix mat_shape: [output_total_size, input_total_size] ([5760, 784])
        size_col_block = mat_shape[1]/kernel_shape[1] # corresponds to in channel
        size_row_block = mat_shape[0]/kernel_shape[0] # corresponds to out channel
        
        edge_idx = list()
        for kidx in kernel_idx:
            # Ex: [1,0,0,4]
            rest_row = int(np.sqrt(mat_shape[1]/kernel_shape[1])) - kernel_shape[2]
            rest_col = int(np.sqrt(mat_shape[1]/kernel_shape[1])) - kernel_shape[3]
            nb_apply_row = rest_row + 1
            nb_apply_col = rest_col + 1
            first_apply = int(np.sqrt(mat_shape[1]/kernel_shape[1]))*kidx[2] + kidx[3]
            total_idx_col = list()
            for idx_col in range(nb_apply_col):
                begin = idx_col*(int(np.sqrt(mat_shape[1]/kernel_shape[1]))) + first_apply
                end = begin + nb_apply_row
                total_idx_col.append( list(range(begin, end)))
            total_idx_col = [int(size_col_block*kidx[1])+item for sublist in total_idx_col for item in sublist]
            total_idx_row = list(int(size_row_block*kidx[0])+np.asarray(range(nb_apply_row*nb_apply_col)))
            total_idx = list(map(list, zip(total_idx_row, total_idx_col)))
            edge_idx += total_idx
        return edge_idx

    q = [process(x) for x in raw_thresholds.split("_")]
    underopt = underopt_edges(0.1, method="magnitude_increase")

    if model.name in ["mnist_lenet", "fashion_mnist_lenet"]:
        mat_shapes = [[5760, 784], [1280, 1440]]
    elif model.name in ["svhn_lenet", "cifar_lenet"]:
        mat_shapes = [[4704, 3072], [1600, 1176]]

    underoptimized_edges = dict()
    j = 0
    c = 0
    for i, layer in enumerate(model.layers):
        if isinstance(layer.func, torch.nn.Conv2d):
            kernel_shape = layer.func.weight.size()
            key = list(underopt.keys())[j]
            underoptimized_edges[model.layer_links[i]] = kernel_to_edge_idx(underopt[key], kernel_shape, mat_shapes[c])
            j += 1
            c += 1
        elif isinstance(layer.func, torch.nn.Linear):
            key = list(underopt.keys())[j]
            underoptimized_edges[model.layer_links[i]] = underopt[key]
            j += 1

    logger.info(f"Keys = {underoptimized_edges.keys()}")
    logger.info(f"Size edges kept = {[len(underoptimized_edges[k]) for k in underoptimized_edges.keys()]}")

    return underoptimized_edges
