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
    from ripser import ripser
except Exception as e:
    logger.warn(e)

from ripser import Rips


def compute_dgm_from_graph_ripser(
    graph: Graph, maxdim: int = 1, n_perm: int = None, debug: bool = False, **kwargs
):
    """
    Use `ripser` tool to compute persistent diagram from graph.
    """
    from scipy import sparse

    adj_mat = sparse.csr_matrix(
        graph.get_adjacency_matrix()
    )  # XXX convert from coo format
    # adj_mat *= -1  # XXX sign correction
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


try:
    from persim import sliced_wasserstein as persim_sw
except Exception as e:
    persim_sw = None


def _prepare_edges_for_diagram_old(graph: Graph) -> typing.Dict:
    all_edges_for_diagrams = graph.get_edge_list_old()

    timing_by_vertex = dict()

    for edge, weight in all_edges_for_diagrams:
        # timing = -weight
        src, tgt = edge
        if weight > timing_by_vertex.get(src, -max_float):
            timing_by_vertex[src] = weight
        if weight > timing_by_vertex.get(tgt, -max_float):
            timing_by_vertex[tgt] = weight

    all_edges_for_diagrams += [
        ([vertex], timing_by_vertex[vertex]) for vertex in timing_by_vertex
    ]

    return all_edges_for_diagrams


def _prepare_edges_for_diagram(graph: Graph) -> typing.Dict:
    timing_by_vertex = {}
    for (src, tgt), weight in graph.get_edge_iter():
        # prepare edge
        yield [src, tgt], weight

        # update vertex data
        if weight > timing_by_vertex.get(src, -max_float):
            timing_by_vertex[src] = weight
        if weight > timing_by_vertex.get(tgt, -max_float):
            timing_by_vertex[tgt] = weight

    # yield vertices
    for vertex in timing_by_vertex:
        yield [vertex], timing_by_vertex[vertex]


def compute_dgm_from_graph(graph: Graph, astuple: bool = True, negate: bool = True):
    all_edges_for_diagrams = _prepare_edges_for_diagram(graph)

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


def sliced_wasserstein_distance_old_version(
    dgm1, dgm2, M=10, verbatim=False, save=False
):
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
    max_diff = np.repeat(0, 10)
    pt1 = list(np.repeat(0, 10))
    pt2 = list(np.repeat(0, 10))
    for _ in range(M):
        v1 = [np.dot(pt1, (np.cos(theta), np.sin(theta))) for pt1 in vec1]
        v2 = [np.dot(pt2, (np.cos(theta), np.sin(theta))) for pt2 in vec2]
        v1 = np.nan_to_num(v1)
        v2 = np.nan_to_num(v2)
        v1, vec1 = (list(t) for t in zip(*sorted(zip(v1, vec1))))
        v2, vec2 = (list(t) for t in zip(*sorted(zip(v2, vec2))))
        # v1.sort()
        # v2.sort()
        diff = np.nan_to_num(np.array(v1) - np.array(v2))
        for i in range(len(max_diff)):
            if max(np.abs(diff)) > max_diff[i]:
                max_diff[i] = max(np.abs(diff))
                idx = np.argmax(np.abs(diff))
                pt1[i] = vec1[idx]
                pt2[i] = vec2[idx]
                break
        sw = sw + s * np.linalg.norm(diff, ord=1)
        theta = theta + s
    if save:
        save_dict = {
            "dgm1": dgm1,
            "dgm2": dgm2,
            "sw": sw,
            "max_diff": max_diff,
            "pt1": pt1,
            "pt2": pt2,
        }
        with open(
            "/Users/m.goibert/Documents/temp/gram_mat/pt_diff_" + verbatim + ".pickle",
            "wb",
        ) as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    # logger.info(f"Max diff = {max_diff} (/ {sw}) and pt1 = {pt1} and pt2 = {pt2}")
    return (1 / np.pi) * sw


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
