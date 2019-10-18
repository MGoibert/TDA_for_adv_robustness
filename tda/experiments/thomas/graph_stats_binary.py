#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np

from tda.graph import Graph
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture, svhn_lenet

from igraph import Graph as IGraph
from networkx.algorithms.centrality import betweenness_centrality, eigenvector_centrality
from networkx.algorithms.centrality.katz import katz_centrality

start_time = time.time()

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--dataset', type=str, default="SVHN")
parser.add_argument('--architecture', type=str, default=svhn_lenet.name)
parser.add_argument('--dataset_size', type=int, default=10)

args, _ = parser.parse_known_args()

logger = logging.getLogger("GraphStats")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)

thresholds = [int(x) for x in args.thresholds.split("_")]


def get_stats(epsilon: float, noise: float) -> typing.List:
    """
    Helper function to get list of embeddings
    """

    weights_per_layer = dict()

    for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=False,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.dataset_size,
            thresholds=thresholds
        ):

        graph: Graph = line[0]

        for i, layer_matrix in enumerate(graph._edge_list):
            if i in weights_per_layer:
                weights_per_layer[i] = np.concatenate([weights_per_layer[i], layer_matrix])
            else:
                weights_per_layer[i] = layer_matrix

    all_weights = list()

    for i in weights_per_layer:
        m = weights_per_layer[i]
        nonzero_m = m[np.where(m > 0)].reshape(-1, 1)
        all_weights.append(nonzero_m)

        q10 = np.quantile(nonzero_m, 0.1)
        q50 = np.quantile(nonzero_m, 0.5)
        q90 = np.quantile(nonzero_m, 0.9)
        print(f"Layer {i} weights {q50} [{q10}; {q90}]")

    all_weights = np.concatenate(all_weights, axis=0)
    q10 = np.quantile(all_weights, 0.1)
    q50 = np.quantile(all_weights, 0.5)
    q90 = np.quantile(all_weights, 0.9)
    q95 = np.quantile(all_weights, 0.95)
    q99 = np.quantile(all_weights, 0.99)
    print(f"All weights {q50} [{q10}; {q90} {q95} {q99}]")


get_stats(epsilon=0.0, noise=0.0)

end_time = time.time()

logger.info(f"Success in {end_time-start_time} seconds")
