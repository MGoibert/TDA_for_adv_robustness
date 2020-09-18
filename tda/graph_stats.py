#!/usr/bin/env python
# coding: utf-8

import typing

import numpy as np

from tda.cache import cached
from tda.dataset.graph_dataset import get_sample_dataset
from tda.graph import Graph
from tda.models import Dataset
from tda.models.architectures import Architecture
from tda.tda_logging import get_logger

logger = get_logger("GraphStats")


#####################
# Fetching datasets #
#####################


@cached
def get_stats_deleted(
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
            logger.info("here1")
            if not isinstance(layer_matrix, np.matrix):
                logger.info(layer_matrix.shape)
                layer_matrix = layer_matrix.todense()
            logger.info("here2")
            if key in weights_per_layer:
                logger.info("here3")
                if not isinstance(weights_per_layer[key], np.matrix):
                    weights_per_layer[key] = weights_per_layer[key].todense()
                logger.info("here4")
                weights_per_layer[key] = np.concatenate(
                    [weights_per_layer[key], layer_matrix]
                )
                logger.info("here5")
            else:
                logger.info("here6")
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


class HistogramQuantile:
    """
    Helper class to compute quantiles without storing too much data.
    We relies on histograms with a max value and precision and assumes that all values are positives.
    """

    def __init__(self, max_value: float = 1e3, precision: int = 1e6):
        self.max_value = max_value
        self.precision = precision
        self.tot_val = 0

        self.counts = dict()

    def get_bucket(self, value: float) -> int:
        if value > self.max_value:
            return self.precision + 1
        else:
            ratio = value / self.max_value
            return int(self.precision * ratio)

    def add_value(self, value: float):
        key = self.get_bucket(value)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.tot_val += 1

    def get_quantiles(self, qs: typing.List[float]) -> typing.List[float]:
        # We assume the quantiles we pass are sorted
        assert qs == sorted(qs)

        covered = 0
        idx = 0

        ret = list()

        for q in qs:
            while q * self.tot_val > covered:
                covered += self.counts.get(idx, 0)
                idx += 1
            ret.append(idx * self.max_value / self.precision)

        return ret


@cached
def get_quantiles_helpers(
    architecture: Architecture, dataset: Dataset, dataset_size: int
) -> typing.Dict:
    assert architecture.is_trained

    quantiles_helpers = dict()

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

            if key not in quantiles_helpers:
                quantiles_helpers[key] = HistogramQuantile()
            helper: HistogramQuantile = quantiles_helpers[key]
            for val in layer_matrix.values:
                helper.add_value(val)

    return quantiles_helpers
