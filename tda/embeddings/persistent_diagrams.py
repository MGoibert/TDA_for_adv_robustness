import typing
import numpy as np
from operator import itemgetter
import logging

from tda.graph import Graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    print("Unable to find dionysus")
    Filtration = None
    Simplex = None
    homology_persistence = None
    init_diagrams = None


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


def sliced_wasserstein_kernel(dgm1, dgm2, M=10):
    #logger.info(f"Sliced Wass. Kernel ")
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
    for i in range(M):
        v1 = [np.dot(pt1, [theta, theta]) for pt1 in vec1]
        v2 = [np.dot(pt2, [theta, theta]) for pt2 in vec2]
        v1.sort()
        v2.sort()
        val = np.nan_to_num(np.asarray(v1) - np.asarray(v2))
        sw = sw + s * np.linalg.norm(val, ord=1)
        theta = theta + s
        #logger.info(f"End Sliced Wass. Kernel")
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
