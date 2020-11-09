import numpy as np
from .covariance_stream_computer import CovarianceStreamComputer, ClassIndex, logger
from typing import Dict, Optional

from .ufunc import v_pinv_sqrt

class NaiveCovarianceStreamComputer(CovarianceStreamComputer):
    """
    Helper object to compute covariance matrices and
    mean of a stream of 1d vectors.
    """

    def __init__(self, min_count_for_sigma: int = 10, layer_idx: int = -1):
        self.sums: Dict[ClassIndex, np.ndarray] = dict()
        self.counts: Dict[ClassIndex, int] = dict()
        self.sigma_sum: Optional[np.ndarray] = None
        self.min_count_for_sigma: int = min_count_for_sigma
        self.layer_idx = layer_idx

    def append(self, x: np.ndarray, clazz: ClassIndex):
        self.sums[clazz] = x.reshape(1, -1) if clazz not in self.sums else self.sums[clazz] + x.reshape(1, -1)
        if self.count > self.min_count_for_sigma:
            c = np.transpose(x - self.mean()) * (x - self.mean())
            self.sigma_sum = c if self.sigma_sum is None else self.sigma_sum + c
        self.counts[clazz] = self.counts.get(clazz, 0) + 1
        # logger.info(
        #    f"CovarianceStreamComputer {self.layer_idx}: {self.count} points (min sigma {self.min_count_for_sigma})"
        # )

    def mean_per_class(self, y: ClassIndex) -> np.ndarray:
        return self.sums[y] / self.counts[y]

    @property
    def count(self):
        return sum(self.counts.values())

    def mean(self) -> np.ndarray:
        return sum(self.sums.values()) / self.count

    @property
    def sigma(self) -> np.ndarray:
        return self.sigma_sum / self.count

    def precision_root(self) -> np.ndarray:
        U, S, V = np.linalg.svd(self.sigma, full_matrices=False, hermitian=True)
        return np.diag(v_pinv_sqrt(S)) @ V
