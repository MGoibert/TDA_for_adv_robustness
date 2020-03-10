#!/usr/bin/env python
# coding: utf-8

import os
import pathlib
import pickle
import typing

import numpy as np

from tda.graph import Graph
from tda.graph_dataset import get_sample_dataset
from tda.tda_logging import get_logger
from tda.models import Dataset
from tda.models.architectures import Architecture
from tda.rootpath import rootpath
from tda.cache import cached

logger = get_logger("GraphStats")


#####################
# Fetching datasets #
#####################


@cached
def get_stats(
    architecture: Architecture, dataset: Dataset, dataset_size: int
) -> (typing.Dict, np.matrix):
    """
    Helper function to get list of embeddings
    """

    assert architecture.is_trained

    weights_per_layer = dict()

    for line in get_sample_dataset(
        epsilon=0.0,
        noise=0.0,
        adv=False,
        archi=architecture,
        dataset_size=dataset_size,
        succ_adv=False,
        dataset=dataset,
        compute_graph=True,
        train=False,
    ):

        graph: Graph = line.graph
        logger.info(
            f"The data point: y = {line.y}, y_pred = {line.y_pred} and adv = {line.y_adv}"
        )
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]
            if not isinstance(layer_matrix, np.matrix):
                layer_matrix = layer_matrix.todense()
            if key in weights_per_layer:
                if not isinstance(weights_per_layer[key], np.matrix):
                    weights_per_layer[key] = weights_per_layer[key].todense()
                weights_per_layer[key] = np.concatenate(
                    [weights_per_layer[key], layer_matrix]
                )
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
        logger.debug(
            f"Link {key} weights [min = {qmin}; "
            f"0.10 = {q10}; 0.25 = {q25}; 0.5 = {q50}; "
            f"0.75 = {q75}; 0.8 = {q80}; 0.9 = {q90}; "
            f"0.95 = {q95}; 0.99 = {q99}; max = {qmax}]"
        )

    return all_weights
