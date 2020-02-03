import numpy as np

from tda.graph import Graph
from tda.logging import get_logger
from numba import njit, prange

logger = get_logger("PersistentDiagrams")
max_float = np.finfo(np.float).max

try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    print("Unable to find dionysus")
    Filtration = None
    Simplex = None
    homology_persistence = None
    init_diagrams = None


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
