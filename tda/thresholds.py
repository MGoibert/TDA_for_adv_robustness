import typing
import logging
import numpy as np

from tda.graph_stats import get_stats
from tda.models import Architecture


def process_thresholds(
        raw_thresholds: str,
        dataset: str,
        architecture: Architecture,
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
        dict_quant = get_stats(
                dataset=dataset,
                architecture=architecture,
                dataset_size=dataset_size,
                epochs=epochs
        )

    for key in thresholds:
        threshold = thresholds[key]
        if 0 < threshold <= 1:
            thresholds[key] = dict_quant[key][threshold]
            logging.info(f"Link {key}: threshold={thresholds[key]} (quantile {threshold})")
        else:
            logging.info(f"Link {key}: threshold={threshold}")

    logging.info(f"Thresholds = {thresholds}")

    return thresholds
