import typing
import os
import logging
import numpy as np

from tda.experiments.thomas.graph_stats_binary import get_stats


def process_thresholds(
        raw_thresholds: str,
        dataset: str,
        architecture: str,
        epochs: int,
        dataset_size: typing.Optional[int] = None
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

    def process(x):
        if x == "inf":
            return np.inf
        return float(x)

    if ";" in raw_thresholds:
        logging.info("Detected new format for thresholds")
        thresholds = {
            (triplet.split(";")[0], triplet.split(";")[1]): triplet.split(";")[2]
            for triplet in  raw_thresholds.split("_")
        }
    else:
        logging.info("Detected legacy format for thresholds")
        thresholds = {
            (i-1, i): process(x)
            for i, x in enumerate(raw_thresholds.split("_"))
        }

    logging.info(f"My received thresholds {thresholds}")

    if any([threshold <= 1 for threshold in thresholds.values()]):
        # In this case, we assume we have threshold as quantiles
        quants_dict_filename = f"stats/{dataset}_{architecture}_{str(epochs)}_epochs.npy"

        if not os.path.exists(quants_dict_filename):
            logging.info(f"Computing weight per layer stats")
            weights, _ = get_stats(epsilon=0.0, noise=0.0, dataset_size=dataset_size)
            quants = np.linspace(0, 1, 1001)
            quants_dict = dict()
            for key in weights:
                weight_layer = weights[key]
                quants_dict[key] = dict()
                for quant in quants:
                    quants_dict[key][quant] = np.quantile(weight_layer, quant)
            np.save(quants_dict_filename, quants_dict)
        dict_quant = np.load(quants_dict_filename, allow_pickle=True).flat[0]

    for key in thresholds:
        threshold = thresholds[key]
        if 0 < threshold <= 1:
            thresholds[key] = dict_quant[key][threshold]
            logging.info(f"Link {key}: threshold={thresholds[key]} (quantile {threshold})")
        else:
            logging.info(f"Link {key}: threshold={threshold}")

    logging.info(f"Thresholds = {thresholds}")

    return thresholds
