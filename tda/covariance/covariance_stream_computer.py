from abc import ABC
import numpy as np
from tda.tda_logging import get_logger

ClassIndex = int

logger = get_logger("CovarianceComputer")


class CovarianceStreamComputer(ABC):
    """
    Helper object to compute covariance matrices and
    mean of a stream of 1d vectors.
    """

    def append(self, x: np.ndarray, clazz: ClassIndex):
        raise NotImplementedError()

    def mean_per_class(self, y: ClassIndex) -> np.ndarray:
        raise NotImplementedError()

    def precision_root(self) -> np.ndarray:
        raise NotImplementedError()

    def precision(self) -> np.ndarray:
        root = self.precision_root()
        return np.transpose(root) @ root
