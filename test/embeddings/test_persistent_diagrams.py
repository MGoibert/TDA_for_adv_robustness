import numpy as np
from tda.embeddings.persistent_diagrams import compute_dgm_from_edges


def test_very_simple_list():

    M = np.zeros((1, 2))
    M[0, 0] = 3
    M[0, 1] = 2

    edge_list = [M]

    ret = compute_dgm_from_edges(edge_list)

    print(ret)