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

    def get_bucket(self, value: float) -> int:
        if value > self.max_value:
            return self.precision + 1
        else:
            ratio = (value - self.min_value) / (self.max_value - self.min_value)
            # Clipping [0, 1]
            ratio = max(min(ratio, 1), 0)
            return int(self.precision * ratio)

    def add_value(self, value: float):
        key = self.get_bucket(value)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.tot_val += 1
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
        compute_graph=True,
        train=False,
    )
    logger.info(f"Got dataset of size {len(dataset)}")

    # Checking min_max
    for line in dataset:
        graph: Graph = line.graph
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]

            min_val, max_val = np.min(layer_matrix.data), np.max(layer_matrix.data)
            old_min, old_max = min_max_vals.get(key, (np.inf, 0))
            min_val = min([min_val, old_min])
            max_val = max([max_val, old_max])
            min_max_vals[key] = (min_val, max_val)

    # Creating quantile helpers
    for line in dataset:
        graph: Graph = line.graph
        for key in graph._edge_dict:
            layer_matrix = graph._edge_dict[key]
            logger.info(f"Line {line.sample_id}: {key}: {layer_matrix.shape}")

            if key not in quantiles_helpers:
                min_val, max_val = min_max_vals[key]
                quantiles_helpers[key] = HistogramQuantile(
                    min_value=0.9 * min_val, max_value=1.1 * max_val, precision=int(1e6)
                )
            helper: HistogramQuantile = quantiles_helpers[key]
            for val in layer_matrix.data:
                helper.add_value(val)

    return quantiles_helpers
