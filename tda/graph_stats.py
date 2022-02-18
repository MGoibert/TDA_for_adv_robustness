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


class HistogramQuantile:
    """
    Helper class to compute quantiles without storing too much data.
    We relies on histograms with a max value and precision and assumes that all values are positives.
    """

    def __init__(
        self, min_value: float = 0.0, max_value: float = 1e3, precision: int = 1e6
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.precision = precision
        self.tot_val = 0

        self.memo = None

        self.counts = dict()

    def get_buckets(self, values: np.ndarray) -> np.ndarray:
        ratios = (values - self.min_value) / (self.max_value - self.min_value)
        # Clipping [0, 1]
        return np.floor(self.precision * np.clip(ratios, 0, 1))

    def add_values(self, values: np.ndarray):
        keys = self.get_buckets(values)
        occ, counters = np.unique(keys, return_counts=True)
        for i in range(len(occ)):
            self.counts[occ[i]] = self.counts.get(occ[i], 0) + counters[i]
            self.tot_val += counters[i]
        self.memo = None

    def get_quantiles(self, qs: typing.List[float]) -> typing.List[float]:
        # We assume the quantiles we pass are sorted
        assert qs == sorted(qs)

        if self.memo is None:
            self.memo = dict()
        else:
            if all([q in self.memo for q in qs]):
                return [self.memo[q] for q in qs]

        covered = 0
        idx = 0

        ret = list()

        for q in qs:
            while q * self.tot_val > covered:
                covered += self.counts.get(idx, 0)
                idx += 1

            quantile = (
                idx * (self.max_value - self.min_value) / self.precision
                + self.min_value
            )
            ret.append(quantile)
            self.memo[q] = quantile

        return ret


@cached
def get_quantiles_helpers(
    architecture: Architecture, dataset: Dataset, dataset_size: int
) -> typing.Dict:
    assert architecture.is_trained

    quantiles_helpers = dict()
    min_max_vals = dict()

    dataset = get_sample_dataset(
        epsilon=0.0,
        noise=0.0,
        adv=False,
        archi=architecture,
        dataset_size=dataset_size,
        succ_adv=False,
        dataset=dataset,
        compute_graph=False,
        train=False,
        dataset_name=dataset.name,
    )
    logger.info(f"Got dataset of size {len(dataset)}")

    size_for_min_max = 10

    # Checking min_max
    for i, line in enumerate(dataset[:size_for_min_max]):
        logger.info(f"Computing min/max sample {i}/{size_for_min_max}")
        graph = Graph.from_architecture_and_data_point(architecture=architecture, x=line.x)
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]

            min_val, max_val = np.min(layer_matrix.data), np.max(layer_matrix.data)
            old_min, old_max = min_max_vals.get(key, (np.inf, 0))
            min_val = min([min_val, old_min])
            max_val = max([max_val, old_max])
            min_max_vals[key] = (min_val, max_val)
        del graph

    logger.info(f"Min-max values are {min_max_vals}")

    # Creating quantile helpers
    for i, line in enumerate(dataset):
        logger.info(f"Computing histograms for quantiles {i}/{len(dataset)}")
        graph = Graph.from_architecture_and_data_point(architecture=architecture, x=line.x)
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]
            # logger.info(f"Line {line.sample_id}: {key}: {layer_matrix.shape}")

            if key not in quantiles_helpers:
                min_val, max_val = min_max_vals[key]
                quantiles_helpers[key] = HistogramQuantile(
                    min_value=0.9 * min_val, max_value=1.1 * max_val, precision=int(1e6)
                )
            helper: HistogramQuantile = quantiles_helpers[key]
            helper.add_values(layer_matrix.data)
        del graph

    return quantiles_helpers
