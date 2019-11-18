import typing
import os
import logging
import numpy as np

from tda.experiments.thomas.graph_stats_binary import get_stats


def process_thresholds(
        raw_thresholds: str,
        dataset: str,
        architecture: str,
        epochs: int
) -> typing.List[float]:
    """
    Compute the actual thresholds to be used from a raw string like

    10_20_13

    OR

    0.1_10000_0.9

    It is assumed that if a threshold is between 0 and 1, it's a QUANTILE
    threshold (which required some stats to be computed on a sample of the
    dataset)

    :param raw_thresholds:
    :param dataset:
    :param architecture:
    :param epochs:
    :return:
    """
    thresholds = [float(x) for x in raw_thresholds.split("_")]

    if any([threshold <= 1 for threshold in thresholds]):
        # In this case, we assume we have threshold as quantiles
        quants_dict_filename = f"stats/{dataset}_{architecture}_{str(epochs)}_epochs.npy"

        if not os.path.exists(quants_dict_filename):
            logging.info(f"Computing weight per layer stats")
            weights, _ = get_stats(epsilon=0.0, noise=0.0)
            quants = np.linspace(0, 1, 1001)
            quants_dict = dict()
            for i, weight_layer in enumerate(weights):
                quants_dict[i] = dict()
                for quant in quants:
                    quants_dict[i][quant] = np.quantile(weight_layer, quant)
            np.save(quants_dict_filename, quants_dict)
        dict_quant = np.load(quants_dict_filename, allow_pickle=True).flat[0]

    for i, threshold in enumerate(thresholds):
        if 0 < threshold <= 1:
            thresholds[i] = dict_quant[i][threshold]
            logging.info(f"Layer {i}: threshold={thresholds[i]} (quantile {threshold})")
        else:
            logging.info(f"Layer {i}: threshold={threshold}")

    logging.info(f"Thresholds = {thresholds}")

    return thresholds
