import typing

import numpy as np

from tda.graph import Graph
from tda.logging import get_logger
from numba import njit, prange

logger = get_logger("PersistentDiagrams")
max_float = np.finfo(np.float).max

try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    logger.warn(e)
    Filtration = None
try:
    from ripser import ripser
except Exception as e:
    logger.warn(e)

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
        graph: Graph,
        astuple: bool = True
):
    all_edges_for_diagrams = graph.get_edge_list()

    timing_by_vertex = dict()

    for edge, weight in all_edges_for_diagrams:
        src, tgt = edge
        if weight < timing_by_vertex.get(src, max_float):
            timing_by_vertex[src] = weight
        if weight < timing_by_vertex.get(tgt, max_float):
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
    dgm = dgms[0]

    if not astuple:
        return dgm
    else:
        ret = list()
        for pt in dgm:
            ret.append((pt.birth, pt.death if not np.isposinf(pt.death) else 2**64))
        return ret


def sliced_wasserstein_distance(dgm1, dgm2, M=10):
    # logger.info(f"Sliced Wass. Kernel ")
    n = len(dgm1) + len(dgm2)
    vec1 = [(0.0, 0.0) for _ in range(n)]
    vec2 = [(0.0, 0.0) for _ in range(n)]
    for i, pt1 in enumerate(dgm1):
        vec1[i] = (pt1[0], pt1[1])
        vec2[i] = ((pt1[0] + pt1[1]) / 2.0, (pt1[0] + pt1[1]) / 2.0)
    for i, pt2 in enumerate(dgm2):
        vec2[i + len(dgm1)] = (pt2[0], pt2[1])
        vec1[i + len(dgm1)] = ((pt2[0] + pt2[1]) / 2.0, (pt2[0] + pt2[1]) / 2.0)
    sw = 0
    theta = -np.pi / 2
    s = np.pi / M
    for _ in range(M):
        v1 = [np.dot(pt1, (np.cos(theta), np.sin(theta))) for pt1 in vec1]
        v2 = [np.dot(pt2, (np.cos(theta), np.sin(theta))) for pt2 in vec2]
        v1.sort()
        v2.sort()
        sw = sw + s * np.linalg.norm(np.nan_to_num(np.array(v1)-np.array(v2)), ord=1)
        theta = theta + s
    return (1 / np.pi) * sw


def sliced_wasserstein_kernel(dgm1, dgm2, M=10, sigma=0.5):
    sw = sliced_wasserstein_distance(dgm1, dgm2, M)
    return np.exp(-sw / (2 * sigma ** 2))
