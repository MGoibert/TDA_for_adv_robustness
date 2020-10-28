from typing import List, Dict

import numpy as np
from sklearn.covariance import LedoitWolf, EmpiricalCovariance, GraphicalLasso
from .ufunc import v_pinv_sqrt

from .covariance_stream_computer import CovarianceStreamComputer, ClassIndex, logger


class SklearnComputer(CovarianceStreamComputer):
    def __init__(self):
        self.data: Dict[ClassIndex, List[np.ndarray]] = dict()
        self.estimator = None
        self.is_fitted = False

    def append(self, x: np.ndarray, clazz: ClassIndex):
        if clazz not in self.data:
            self.data[clazz] = list()
        self.data[clazz].append(x.reshape(1, -1))
        self.is_fitted = False

    def fit(self):
        if not self.is_fitted:
            all_x = [
                elem.reshape(-1) for a_list in self.data.values() for elem in a_list
            ]
            self.estimator.fit(all_x)
            self.is_fitted = True

    def mean_per_class(self, clazz: ClassIndex) -> np.ndarray:
        return np.mean(self.data[clazz], 0)

    def precision_root(self) -> np.ndarray:
        self.fit()
        U, S, V = np.linalg.svd(self.estimator.covariance_, full_matrices=False, hermitian=True)
        return np.diag(v_pinv_sqrt(S)) @ V


class LedoitWolfComputer(SklearnComputer):
    def __init__(self):
        super().__init__()
        self.estimator = LedoitWolf()


class EmpiricalSklearnComputer(SklearnComputer):
    def __init__(self):
        super().__init__()
        self.estimator = EmpiricalCovariance()


class GraphicalLassoComputer(SklearnComputer):
    def fit(self):
        if not self.is_fitted:
            all_x = [
                elem.reshape(-1) for a_list in self.data.values() for elem in a_list
            ]
            e = None
            for alpha in np.logspace(-1, 5, 10):
                try:
                    self.estimator = GraphicalLasso(assume_centered=False, alpha=alpha)
                    self.estimator.fit(all_x)
                    self.is_fitted = True
                    return
                except Exception as e:
                    logger.error(f"Graphical lasso failed with alpha={alpha}")
            raise e
