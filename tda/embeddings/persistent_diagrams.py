import typing

import numpy as np

from tda.graph import Graph
from tda.logging import get_logger

logger = get_logger("PersistentDiagrams")

try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    print("Unable to find dionysus")
    Filtration = None
    Simplex = None
    homology_persistence = None
    init_diagrams = None

from ripser import Rips


def compute_dgm_from_graph_ripser(
        graph: Graph,
        maxdim: int=1,
        n_perm: int=None,
        debug: bool=False,
        **kwargs):
    """
    Use `ripser` tool to compute persistent diagram from graph.
    """
    from scipy import sparse
    adj_mat = sparse.csr_matrix(graph.get_adjacency_matrix()) # XXX convert from coo format
    adj_mat *= -1  # XXX sign correction
    rips = Rips(maxdim=maxdim, n_perm=n_perm, **kwargs)
    import time
    if False:
        t0 = time.time()
        dist_mat = sparse.csgraph.floyd_warshall(adj_mat, directed=True)
        print(adj_mat.shape, adj_mat.count_nonzero())
        print("Spent %gs computing dist_mat" % (time.time() - t0))
    else:
        dist_mat = adj_mat
    t0 = time.time()    
    dgms = rips.fit_transform(dist_mat, distance_matrix=True)
    print("Spent %gs computing persistence diagrams" % (time.time() - t0))    
    print(list(map(len, dgms)))
    if debug:
        import matplotlib.pyplot as plt
        rips.plot()
        plt.show()
    return np.vstack(dgms)


def compute_dgm_from_graph(
        graph: Graph
):
    all_edges_for_diagrams = graph.get_edge_list()

    timing_by_vertex = dict()

    for edge, weight in all_edges_for_diagrams:
        src, tgt = edge
        if weight < timing_by_vertex.get(src, np.inf):
            timing_by_vertex[src] = weight
        if weight < timing_by_vertex.get(tgt, np.inf):
            timing_by_vertex[tgt] = weight

    all_edges_for_diagrams += [
        ([vertex], timing_by_vertex[vertex])
        for vertex in timing_by_vertex
    ]

    # Dionysus computations (persistent diagrams)
    # logger.info(f"Before filtration")
    f = Filtration()
    for vertices, timing in all_edges_for_diagrams:
        f.append(Simplex(vertices, timing))
    f.sort()
    m = homology_persistence(f)
    dgms = init_diagrams(m, f)

    return dgms[0]


def get_birth_death(pt) -> typing.Tuple[float, float]:
    if hasattr(pt, "birth"):
        return pt1.birth, pt1.death
    else:
        return pt

def sliced_wasserstein_kernel(dgm1, dgm2, M=10):
    # logger.info(f"Sliced Wass. Kernel ")
    vec1 = []
    vec2 = []
    dgm1 = map(get_birth_death, dgm1)
    dgm2 = map(get_birth_death, dgm2)
    for birth, death in dgm1:
        vec1.append([birth, death])
        vec2.append([(birth + death) / 2.0, (birth + death) / 2.0])
    for beath, death in dgm2:
        vec2.append([birth, death])
        vec1.append([(birth + death) / 2.0, (birth + death) / 2.0])
    sw = 0
    theta = -np.pi / 2
    s = np.pi / M
    for _ in range(M):
        v1 = [np.dot(pt1, [theta, theta]) for pt1 in vec1]
        v2 = [np.dot(pt2, [theta, theta]) for pt2 in vec2]
        v1.sort()
        v2.sort()
        val = np.nan_to_num(np.asarray(v1) - np.asarray(v2))
        sw = sw + s * np.linalg.norm(val, ord=1)
        theta = theta + s
        # logger.info(f"End Sliced Wass. Kernel")
        # print("Run :", i, " and sw =", (1/np.pi)*sw)
    return (1 / np.pi) * sw


def sliced_wasserstein_gram(DGM1, DGM2, M=10, sigma=0.1):
    n = len(DGM1)
    m = len(DGM2)
    gram = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            sw = sliced_wasserstein_kernel(DGM1[i], DGM2[j], M=M)
            gram[i, j] = np.exp(-sw / (2 * sigma ** 2))
    return gram
