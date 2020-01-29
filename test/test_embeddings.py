import numpy as np
import torch

from tda.embeddings import get_gram_matrix, KernelType, compute_dgm_from_graph, get_gram_matrix_legacy
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import SoftMaxLayer, LinearLayer


def test_euclidean_gram():
    embed_in = [np.random.randn(50, 1) for _ in range(10)]
    embed_out = [np.random.randn(50, 1) for _ in range(5)]

    M = get_gram_matrix(
        kernel_type=KernelType.Euclidean,
        embeddings_in=embed_in,
        embeddings_out=embed_out)

    assert M.shape == (10, 5)


def test_sliced_wasserstein_gram_matrix(benchmark):
    simple_archi: Architecture = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2),
            LinearLayer(2, 10),
            SoftMaxLayer()
        ])

    embeddings = list()

    for val in np.random.randint(1, 100, 100):
        ex = torch.ones(4) * val
        g = Graph.from_architecture_and_data_point(simple_archi, ex, thresholds=dict())
        dgm = compute_dgm_from_graph(g)
        embeddings.append(dgm)

    def compute_matrix():
        return get_gram_matrix(
            kernel_type=KernelType.SlicedWasserstein,
            embeddings_in=embeddings,
            embeddings_out=embeddings,
            params={"M": 20, "sigma": 0.5}
        )

    matrix = benchmark(compute_matrix)

    legacy_matrix = get_gram_matrix_legacy(
            kernel_type=KernelType.SlicedWasserstein,
            embeddings_in=embeddings,
            embeddings_out=embeddings,
            params={"M": 20, "sigma": 0.5}
        )

    assert np.isclose(np.linalg.norm(matrix), 99.9999999)
    assert np.isclose(np.linalg.norm(matrix-legacy_matrix), 0.0)
