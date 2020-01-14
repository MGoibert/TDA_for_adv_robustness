#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pathlib
import pickle
import time
import typing

import numpy as np

from tda.graph import Graph
from tda.graph_dataset import get_graph_dataset
from tda.models.architectures import get_architecture
from tda.models.architectures import mnist_lenet, Architecture
from tda.rootpath import rootpath
from tda.logging import get_logger

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--thresholds', type=str, default="0_0_0_0_0_0_0")
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_lenet.name)
parser.add_argument('--train_noise', type=float, default=0.0)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--num_iter', type=int, default=10)
parser.add_argument('--visualize_adj_mat', type=float, default=0)

args, _ = parser.parse_known_args()

logger = get_logger("GraphStats")

#####################
# Fetching datasets #
#####################


def get_stats(
        epochs: int,
        architecture: Architecture,
        dataset: str,
        dataset_size: int,
        train_noise: float = 0.0
) -> (typing.Dict, np.matrix):
    """
    Helper function to get list of embeddings
    """
    base_path = f"{rootpath}/stats/"+"/".join(sorted([f"{k}={str(v)}" for (k, v) in locals().items()]))
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    quants_dict_filename = f"{base_path}/stats.pickle"

    if os.path.exists(quants_dict_filename):
        logger.info(f"Loading stats from file {quants_dict_filename}")
        with open(quants_dict_filename, "rb") as f:
            return pickle.load(f)

    logger.info(f"Cannot find {quants_dict_filename}. Recomputing stats...")

    weights_per_layer = dict()

    for line in get_graph_dataset(
            # num_epochs=epochs,
            epsilon=0.0,
            noise=0.0,
            adv=False,
            # retain_data_point=False,
            architecture=architecture,
            source_dataset_name=dataset,
            dataset_size=dataset_size,
            thresholds=None,
            only_successful_adversaries=False,
            train_noise=train_noise,
            use_sigmoid=False
        ):

        graph: Graph = line.graph
        logger.info(f"The data point: y = {line.y}, y_pred = {line.y_pred} and adv = {line.y_adv}")
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]
            if not isinstance(layer_matrix, np.matrix):
                layer_matrix = layer_matrix.todense()
            if key in weights_per_layer:
                if not isinstance(weights_per_layer[key], np.matrix):
                    weights_per_layer[key] = weights_per_layer[key].todense()
                weights_per_layer[key] = np.concatenate([weights_per_layer[key], layer_matrix])
            else:
                weights_per_layer[key] = layer_matrix

    all_weights = dict()

    for key in weights_per_layer:
        m = weights_per_layer[key]
        nonzero_m = m[np.where(m > 0)].reshape(-1, 1)
        logger.info(f"Number of edges > 0 in link {key}: {len(nonzero_m)}")
        all_weights[key] = nonzero_m

        qmin = min(nonzero_m)
        q10 = np.quantile(nonzero_m, 0.1)
        q25 = np.quantile(nonzero_m, 0.25)
        q50 = np.quantile(nonzero_m, 0.5)
        q75 = np.quantile(nonzero_m, 0.75)
        q80 = np.quantile(nonzero_m, 0.8)
        q90 = np.quantile(nonzero_m, 0.9)
        q95 = np.quantile(nonzero_m, 0.95)
        q99 = np.quantile(nonzero_m, 0.99)
        qmax = max(nonzero_m)
        print(f"Link {key} weights [min = {qmin}; "
              f"0.10 = {q10}; 0.25 = {q25}; 0.5 = {q50}; "
              f"0.75 = {q75}; 0.8 = {q80}; 0.9 = {q90}; "
              f"0.95 = {q95}; 0.99 = {q99}; max = {qmax}]")

    quants = np.linspace(0, 1, 1001)
    quants_dict = dict()
    for key in all_weights:
        weight_layer = all_weights[key]
        quants_dict[key] = dict()
        for quant in quants:
            quants_dict[key][quant] = np.quantile(weight_layer, quant)

    with open(quants_dict_filename, "wb") as f:
        pickle.dump(quants_dict, f)

    return quants_dict


if __name__ == '__main__':

    start_time = time.time()

    quantiles = get_stats(
        dataset_size=args.dataset_size,
        architecture=get_architecture(args.architecture),
        dataset=args.dataset,
        epochs=args.epochs,
        train_noise=args.train_noise
    )

    logger.info(quantiles.keys())
    end_time = time.time()

    logger.info(f"Success in {end_time-start_time} seconds")
