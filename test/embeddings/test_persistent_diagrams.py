import fwg
import numpy as np
import torch

from tda.embeddings import get_gram_matrix, KernelType
from tda.embeddings.persistent_diagrams import (
    compute_dgm_from_graph,
    sliced_wasserstein_distance_matrix,
    compute_dgm_from_edges,
)
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import LinearLayer, SoftMaxLayer

simple_archi: Architecture = Architecture(
    preprocess=lambda x: x,
    layers=[LinearLayer(4, 3), LinearLayer(3, 2), LinearLayer(2, 10), SoftMaxLayer()],
)

ex1 = torch.ones(4) * 42
ex2 = torch.ones(4) * 37

simple_archi.build_matrices()

g1 = Graph.from_architecture_and_data_point(simple_archi, ex1)
g2 = Graph.from_architecture_and_data_point(simple_archi, ex2)

dgm1 = compute_dgm_from_graph(g1, astuple=False)
dgm2 = compute_dgm_from_graph(g2, astuple=False)

dgm1_tuple = compute_dgm_from_graph(g1, astuple=True)
dgm2_tuple = compute_dgm_from_graph(g2, astuple=True)


def test_simple_edge_list():

    edge_list = [((0, 1), np.float64(3.0)), ((0, 2), np.float64(2.0))]

    ret = compute_dgm_from_edges(edge_list)

    print(ret)


def test_get_gram():
    embeddings = [list(dgm1_tuple), list(dgm2_tuple), [(1, 3), (2, 4), (5, 8)]]

    m = get_gram_matrix(
        embeddings_in=embeddings,
        embeddings_out=embeddings,
        kernel_type=KernelType.SlicedWasserstein,
        params=[{"M": 10, "sigma": 0.1}],
    )

    print(m)


def test__wasserstein_distances_c_vs_python():
    embeddings = [
        [(2.0, 4.0), (4.0, 8.0)],
        [(1.0, 2.0), (8.0, 20.0), (34.0, 90.0)],
        [(2.0, 4.0), (4.0, 8.0)],
    ]

    print(embeddings)

    c_gram = fwg.fwd(embeddings, embeddings, 50)

    print("c++", c_gram)

    python_gram = sliced_wasserstein_distance_matrix(
        embeddings, embeddings, 50, software="builtin"
    )

    print("python", python_gram)

    persim_python_gram = sliced_wasserstein_distance_matrix(
        embeddings, embeddings, 50, software="persim"
    )

    print("persim", persim_python_gram)

    assert np.isclose(np.linalg.norm(c_gram - python_gram), 0.0, atol=1e-7)
    assert np.isclose(np.linalg.norm(c_gram - persim_python_gram), 0.0, atol=1e-7)
