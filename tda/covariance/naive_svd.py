from typing import Dict

import numpy as np

from .covariance_stream_computer import CovarianceStreamComputer, ClassIndex, logger
from .ufunc import v_pinv


class NaiveSVDCovarianceStreamComputer(CovarianceStreamComputer):
    def __init__(self):
        self.data: Dict[ClassIndex, np.ndarray] = dict()

    def append(self, x: np.ndarray, clazz: ClassIndex):
        if clazz not in self.data:
            self.data[clazz] = x.reshape(1, -1)
        else:
            self.data[clazz] = np.vstack([self.data[clazz], x.reshape(1, -1)])

    def mean_per_class(self, clazz: ClassIndex) -> np.ndarray:
        return np.mean(self.data[clazz], 0)

    def precision_root(self) -> np.ndarray:
        all_x = np.vstack(list(self.data.values()))
        arr = (all_x - np.mean(all_x, 0)) / np.sqrt(np.shape(all_x)[0])
        logger.info(f"Computing SVD for matrix of shape {arr.shape}")
        U, S, V = np.linalg.svd(arr, full_matrices=False)
        logger.info(f"SVD done !")
        del all_x
        del arr
        root = np.diag(v_pinv(S)) @ V
        logger.info(f"Root size is {root.nbytes} bytes")
        return root
