import numpy as np
import torch
import fwg

from tda.embeddings import get_gram_matrix, KernelType
from tda.embeddings.persistent_diagrams import compute_dgm_from_graph, \
    sliced_wasserstein_distance, sliced_wasserstein_distance_matrix
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import LinearLayer, SoftMaxLayer
import dionysus
from persim import sliced_wasserstein as persim_sw, wasserstein as persim_w

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


def test_sliced_wassertstein_kernel():
    approx_distance = sliced_wasserstein_distance(dgm1_tuple, dgm2_tuple, M=100)
    real_distance = dionysus.wasserstein_distance(dgm1, dgm2, q=1)

    gram = fwg.fwd(
        [dgm1_tuple, dgm2_tuple],
        [dgm1_tuple, dgm2_tuple],
        100
    )

    print(gram)
    print(approx_distance)
    print(real_distance)


def test_gram():
    embeddings = [
        list(dgm1_tuple),
        list(dgm2_tuple),
        [(1, 3), (2, 4), (5, 8)]
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


def test_exact_wasserstein_gram():
    embeddings = [
        [(2, 4), (4, 8)],
        [(1, 2), (8, 20), (34, 90)],
        [(2, 4), (4, 8)]
    ]

    distances_2 = [
        [sliced_wasserstein_distance(
            embeddings[i], embeddings[j], M=50
        )
            for j in range(3)]
        for i in range(3)
    ]

    print(distances_2)

    distances_3 = [
        [(persim_sw(
            np.array(np.nan_to_num(embeddings[i])), np.nan_to_num(np.array(embeddings[j])), M=50
        ),
            persim_w(np.array(np.nan_to_num(embeddings[i])), np.nan_to_num(np.array(embeddings[j])))
        )
            for j in range(3)]
        for i in range(3)
    ]

    print(distances_3)


def test_fast_wasserstein_gram_c_version(benchmark):
    embeddings = [
        [(2.0, 4.0), (4.0, 8.0)],
        [(1.0, 2.0), (8.0, 20.0), (34.0, 90.0)],
        [(2.0, 4.0), (4.0, 8.0)]
    ]

    print(embeddings)

    def b():
        gram = np.reshape(fwg.fwd(
            embeddings,
            embeddings,
            50
        ), (len(embeddings), len(embeddings)))
        return gram

    print(b())


def test_fast_wasserstein_gram_python_version(benchmark):
    embeddings = [
        [(2.0, 4.0), (4.0, 8.0)],
        [(1.0, 2.0), (8.0, 20.0), (34.0, 90.0)],
        [(2.0, 4.0), (4.0, 8.0)]
    ]

    print(embeddings)

    def b():
        gram = sliced_wasserstein_distance_matrix(
            embeddings,
            embeddings,
            50
        )
        return gram

    print(b())
