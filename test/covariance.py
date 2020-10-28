import numpy as np
from typing import List

from tda.covariance import (
    NaiveSVDCovarianceStreamComputer,
    NaiveCovarianceStreamComputer,
    LedoitWolfComputer,
    CovarianceStreamComputer,
    EmpiricalSklearnComputer,
    GraphicalLassoComputer,
)


def test_methods():

    np.random.seed(37920284)

    dim = 5
    nb_samples = 10000

    activations = [2 * np.random.randn(dim) + 5 for _ in range(nb_samples)]

    classes = [i % 3 for i in range(nb_samples)]

    computers: List[CovarianceStreamComputer] = [
        NaiveCovarianceStreamComputer(min_count_for_sigma=5000),
        NaiveSVDCovarianceStreamComputer(),
        LedoitWolfComputer(),
        EmpiricalSklearnComputer(),
        GraphicalLassoComputer(),
    ]

    for computer in computers:
        for activ, clazz in zip(activations, classes):
            computer.append(activ, clazz)

    expected_mean = np.mean(activations, 0)

    arr = np.array(activations) - np.mean(activations, 0)
    true_empirical = np.linalg.pinv(
        np.transpose(arr) @ arr / nb_samples, hermitian=True
    )

    for computer in computers:
        print("\n" + computer.__class__.__name__)
        print(
            f"Norm to true empirical {np.linalg.norm(computer.precision()-true_empirical)}"
        )
        print("---")
