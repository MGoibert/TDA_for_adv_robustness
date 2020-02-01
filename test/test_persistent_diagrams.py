import numpy as np
import torch
from ripser import Rips

from tda.embeddings import get_gram_matrix, KernelType
from tda.embeddings.persistent_diagrams import compute_dgm_from_graph, \
    sliced_wasserstein_kernel, sliced_wasserstein_kernel_legacy, fast_wasserstein_gram
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import LinearLayer, SoftMaxLayer

simple_archi: Architecture = Architecture(
    preprocess=lambda x: x,
    layers=[
        LinearLayer(4, 3),
        LinearLayer(3, 2),
        LinearLayer(2, 10),
        SoftMaxLayer()
    ])

ex1 = torch.ones(4) * 42
ex2 = torch.ones(4) * 37

g1 = Graph.from_architecture_and_data_point(simple_archi, ex1, thresholds=dict())
g2 = Graph.from_architecture_and_data_point(simple_archi, ex2, thresholds=dict())

############
# Dionysus #
############

dgm1 = compute_dgm_from_graph(g1, astuple=False)
dgm2 = compute_dgm_from_graph(g2, astuple=False)

dgm1_tuple = compute_dgm_from_graph(g1, astuple=True)
dgm2_tuple = compute_dgm_from_graph(g2, astuple=True)


def test_sliced_wassertstein_kernel_legacy(benchmark):
    def compute_dgm():
        return sliced_wasserstein_kernel_legacy(dgm1, dgm2)

    k_dionysus = benchmark(compute_dgm)
    assert np.isclose(k_dionysus, 7.450580596923829e-10, rtol=1e-12)


def test_sliced_wassertstein_kernel(benchmark):
    def compute_dgm():
        return sliced_wasserstein_kernel(dgm1_tuple, dgm2_tuple)

    k_dionysus = benchmark(compute_dgm)
    assert np.isclose(k_dionysus, 7.450580596923829e-10, rtol=1e-12)


def test_gram():
    embeddings = [
        dgm1_tuple,
        dgm2_tuple,
        ((1, 3), (2, 4), (5, 8))
    ]

    m = get_gram_matrix(
        embeddings_in=embeddings,
        embeddings_out=embeddings,
        kernel_type=KernelType.SlicedWasserstein,
        params={
            'M': 10,
            'sigma': 0.1
        }
    )

    print(m)


def test_fast_wasserstein_gram(benchmark):
    embeddings = [
        dgm1_tuple,
        dgm2_tuple,
        ((1, 3), (2, 4), (5, 8))
    ]

    def b():
        gram = fast_wasserstein_gram(
            embeddings_in=embeddings * 10,
            embeddings_out=embeddings * 50,
            M=10,
            sigma=0.1
        )
        return gram

    benchmark(b)
