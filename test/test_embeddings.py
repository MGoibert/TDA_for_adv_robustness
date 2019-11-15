import numpy as np
from tda.embeddings import get_gram_matrix, KernelType


def test_euclidean_gram():
    embed_in = [np.random.randn(50,1) for _ in range(10)]
    embed_out = [np.random.randn(50, 1) for _ in range(5)]

    M = get_gram_matrix(
        kernel_type=KernelType.Euclidean,
        embeddings_in=embed_in,
        embeddings_out=embed_out)

    assert M.shape == (10, 5)

