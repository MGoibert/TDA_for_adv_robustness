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
        # Keeping only a tuple with the list of the points
        ret = list((pt.birth, pt.death) for pt in dgm)
        return ret


def sliced_wasserstein_kernel_legacy(dgm1, dgm2, M=10):
    # logger.info(f"Sliced Wass. Kernel ")
    vec1 = []
    vec2 = []
    for pt1 in dgm1:
        vec1.append([pt1.birth, pt1.death])
        vec2.append([(pt1.birth + pt1.death) / 2.0, (pt1.birth + pt1.death) / 2.0])
    for pt2 in dgm2:
        vec2.append([pt2.birth, pt2.death])
        vec1.append([(pt2.birth + pt2.death) / 2.0, (pt2.birth + pt2.death) / 2.0])
    sw = 0
    theta = -np.pi / 2
    s = np.pi / M
    for _ in range(M):
        v1 = [np.dot(pt1, (theta, theta)) for pt1 in vec1]
        v2 = [np.dot(pt2, (theta, theta)) for pt2 in vec2]
        v1.sort()
        v2.sort()
        val = np.asarray(v1) - np.asarray(v2)
        val = np.nan_to_num(val)
        sw = sw + s * np.linalg.norm(val, ord=1)
        theta = theta + s
        # logger.info(f"End Sliced Wass. Kernel")
        # print("Run :", i, " and sw =", (1/np.pi)*sw)
    return (1 / np.pi) * sw


def sliced_wasserstein_kernel(dgm1, dgm2, M=10):
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
        v1 = [np.dot(pt1, (theta, theta)) for pt1 in vec1]
        v2 = [np.dot(pt2, (theta, theta)) for pt2 in vec2]
        v1.sort()
        v2.sort()
        norm1 = 0
        for l in range(n):
            raw_diff = v1[l] - v2[l]
            if np.isposinf(raw_diff) or np.isneginf(raw_diff):
                norm1 += max_float
            elif not np.isnan(raw_diff):
                norm1 += abs(raw_diff)
        sw = sw + s * norm1
        theta = theta + s
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


def fast_wasserstein_gram(
        embeddings_in,
        embeddings_out,
        M,
        sigma
):
    n = len(embeddings_in)
    m = len(embeddings_out)
    gram = np.array([[0.0 for _ in range(m)] for _ in range(n)])

    for i in range(n):
        for j in range(m):
            dgm1 = embeddings_in[i]
            dgm2 = embeddings_out[j]
            # logger.info(f"Sliced Wass. Kernel ")
            u = len(dgm1) + len(dgm2)
            vec1 = [(0.0, 0.0) for _ in range(u)]
            vec2 = [(0.0, 0.0) for _ in range(u)]
            for k, pt1 in enumerate(dgm1):
                vec1[k] = (pt1[0], pt1[1])
                vec2[k] = ((pt1[0] + pt1[1]) / 2.0, (pt1[0] + pt1[1]) / 2.0)
            for k, pt2 in enumerate(dgm2):
                vec2[k + len(dgm1)] = (pt2[0], pt2[1])
                vec1[k + len(dgm1)] = ((pt2[0] + pt2[1]) / 2.0, (pt2[0] + pt2[1]) / 2.0)
            sw = 0
            theta = -np.pi / 2
            s = np.pi / M
            for k in range(M):
                v1 = [np.dot(pt1, (theta, theta)) for pt1 in vec1]
                v2 = [np.dot(pt2, (theta, theta)) for pt2 in vec2]
                v1.sort()
                v2.sort()
                val = np.asarray(v1) - np.asarray(v2)
                val[np.isnan(val)] = 0.0
                # val = np.nan_to_num(np.asarray(v1) - np.asarray(v2))
                sw = sw + s * np.linalg.norm(val, ord=1)
                theta = theta + s
                # logger.info(f"End Sliced Wass. Kernel")
                # print("Run :", i, " and sw =", (1/np.pi)*sw)
            gram[i, j] = np.exp(-(1 / np.pi) * sw / (2 * sigma ** 2))

    return gram
