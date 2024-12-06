import numpy as np
import torch

from tda.embeddings import (
    get_gram_matrix,
    KernelType,
    compute_dgm_from_graph,
    get_gram_matrix_legacy,
)
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import SoftMaxLayer, LinearLayer


def test_euclidean_gram():
    embed_in = [np.random.randn(50, 1) for _ in range(10)]
    embed_out = [np.random.randn(50, 1) for _ in range(5)]

    M = get_gram_matrix(
        kernel_type=KernelType.Euclidean,
        embeddings_in=embed_in,
        embeddings_out=embed_out,
    )[0]

    M_legacy = get_gram_matrix_legacy(
        kernel_type=KernelType.Euclidean,
        embeddings_in=embed_in,
        embeddings_out=embed_out,
    )

    print(M_legacy - M)

    assert np.isclose(np.linalg.norm(M - M_legacy), 0)

    assert M.shape == (10, 5)


def test_sliced_wasserstein_gram_matrix(benchmark):
    simple_archi: Architecture = Architecture(
        preprocess=lambda x: x, layers=[LinearLayer(4, 3), SoftMaxLayer()]
    )
    simple_archi.build_matrices()

    embeddings = list()

    for val in np.random.randint(1, 100, 100):
        idx = np.random.randint(0, 4)
        z = np.zeros(4)
        z[idx] = val
        ex = torch.from_numpy(z)
        g = Graph.from_architecture_and_data_point(simple_archi, ex)
        dgm = compute_dgm_from_graph(g)
        embeddings.append(dgm)

    def compute_matrix():
        return get_gram_matrix(
            kernel_type=KernelType.SlicedWasserstein,
            embeddings_in=embeddings,
            embeddings_out=embeddings,
            params=[{"M": 50, "sigma": 0.5}],
        )[0]

    legacy_matrix = get_gram_matrix_legacy(
        kernel_type=KernelType.SlicedWasserstein,
        embeddings_in=embeddings,
        embeddings_out=embeddings,
        params={"M": 50, "sigma": 0.5},
    )

    print(legacy_matrix)

    matrix = benchmark(compute_matrix)

    print(matrix)

    assert np.isclose(np.linalg.norm(matrix - legacy_matrix), 0.0)
