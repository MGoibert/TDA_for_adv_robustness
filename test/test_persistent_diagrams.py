import numpy as np
import torch
from ripser import Rips

from tda.embeddings.persistent_diagrams import compute_dgm_from_graph, sliced_wasserstein_kernel, _helper_fast, _helper_slow
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

dgm1 = compute_dgm_from_graph(g1)
dgm2 = compute_dgm_from_graph(g2)


def test_sliced_wassertstein_kernel(benchmark):

    def compute_dgm():
        return sliced_wasserstein_kernel(dgm1, dgm2)

    k_dionysus = benchmark(compute_dgm)
    assert np.isclose(k_dionysus, 7.450580596923829e-10, rtol=1e-12)

    rips = Rips(maxdim=1, coeff=2)

    dgm1_ripser = rips.fit_transform(g1.get_adjacency_matrix(), distance_matrix=True)
    dgm1_ripser_alt = rips.fit_transform(-g1.get_adjacency_matrix(), distance_matrix=False)

    print("Dionysus")
    print(dgm1)
    for pt in dgm1:
        print(pt)

    print("----")
    print("Ripser")
    print(dgm1_ripser)
    print("##")
    print(dgm1_ripser_alt)


def test_helper_slow(benchmark):
    a = ((1.0, 34.0), (2.2, 33.2)) * 10
    b = ((2.9, 22.6), (4.5, 46.3)) * 10

    def f():
        return _helper_slow(a, b, 10)

    result = benchmark(f)

    assert np.isclose(result, 195.56414268596455)


def test_helper_fast(benchmark):
    a = ((1.0, 34.0), (2.2, 33.2)) * 10
    b = ((2.9, 22.6), (4.5, 46.3)) * 10

    def f():
        return _helper_fast(a, b, 10)

    result = benchmark(f)

    assert np.isclose(result, _helper_slow(a, b, 10))
    assert np.isclose(result, 195.56414268596455)


