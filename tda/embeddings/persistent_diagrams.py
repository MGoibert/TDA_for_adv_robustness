import typing

import numpy as np
import pickle

from tda.graph import Graph
from tda.tda_logging import get_logger
import typing

logger = get_logger("PersistentDiagrams")
max_float = np.finfo(np.float).max

try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    logger.warn(e)
    Filtration = None

try:
    from persim import sliced_wasserstein as persim_sw
except Exception as e:
    persim_sw = None


def _prepare_edges_for_diagram(edge_list: typing.List):
    """
    Enrich the edge list with the vertex and find their birth date
    """

    timing_by_vertex = dict()

    for edge, weight in edge_list:
        # timing = -weight
        src, tgt = edge
        if weight > timing_by_vertex.get(src, -max_float):
            timing_by_vertex[src] = weight
        if weight > timing_by_vertex.get(tgt, -max_float):
            timing_by_vertex[tgt] = weight

    edge_list += [([vertex], timing_by_vertex[vertex]) for vertex in timing_by_vertex]


def compute_dgm_from_graph(graph: Graph, astuple: bool = True, negate: bool = True):
    return compute_dgm_from_edges(graph.get_edge_list(), astuple=astuple, negate=negate)


def compute_dgm_from_edges(
    all_edges_for_diagrams, astuple: bool = True, negate: bool = True
):
    _prepare_edges_for_diagram(all_edges_for_diagrams)

    # Dionysus computations (persistent diagrams)
    # logger.info(f"Before filtration")

    f = Filtration()
    for vertices, weight in all_edges_for_diagrams:
        if negate:
            timing = -weight.round(3)
        else:
            timing = weight.round(3)
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
            # ret.append((pt.birth, pt.death if not np.isposinf(pt.death) else 2**64))
            ret.append((round(pt.birth, 2), round(pt.death, 2)))
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
        sw = sw + s * np.linalg.norm(np.nan_to_num(np.array(v1) - np.array(v2)), ord=1)
        theta = theta + s
    return (1 / np.pi) * sw


def sliced_wasserstein_kernel(dgm1, dgm2, M=10, sigma=0.5):
    sw = sliced_wasserstein_distance(dgm1, dgm2, M)
    return np.exp(-sw / (2 * sigma ** 2))


def sliced_wasserstein_distance_matrix(
    embeddings_in: typing.List, embeddings_out: typing.List, M: int, software="builtin"
):
    n = len(embeddings_in)
    m = len(embeddings_out)
    ret = np.zeros(n * m)

    for i in range(n):
        for j in range(m):
            if software == "builtin":
                ret[i * n + j] = sliced_wasserstein_distance(
                    embeddings_in[i], embeddings_out[j], M
                )
            elif software == "persim":
                ret[i * n + j] = persim_sw(
                    np.array(embeddings_in[i]), np.array(embeddings_out[j]), M
                )
    return np.reshape(ret, (n, m))
