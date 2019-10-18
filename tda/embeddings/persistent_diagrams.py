import typing
import numpy as np
from operator import itemgetter
import logging

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


def compute_dgm_from_edges(
        edge_list: typing.List
):
    """
    Create the simplicial complexes using dionysus
    Fast implementation but "by hand"
    """

    vec = []
    shape = np.cumsum([edge_list[key].shape[1] for key in range(len(edge_list))])

    shape = np.insert(shape, 0, 0)
    shape = np.insert(shape, len(shape), shape[len(shape) - 1] + edge_list[len(shape) - 2].shape[0])

    for layer_idx in range(len(edge_list)):
        logger.debug(f"Layer nb {layer_idx}")
        # Adding the edges
        row, col = np.meshgrid(np.arange(shape[layer_idx], shape[layer_idx + 1]),
                               np.arange(shape[layer_idx + 1], shape[layer_idx + 2]))
        ind = np.where(edge_list[layer_idx].ravel() > 0.0)
        if len(ind) > 1:
            ind = ind[1]
        else:
            ind = ind[0]
        if len(edge_list[layer_idx].ravel()[0].shape) >= 1:
            table = np.vstack((np.asarray(edge_list[layer_idx].ravel())[0][ind], row.ravel()[ind], col.ravel()[ind])).T
        else:
            table = np.vstack((np.asarray(edge_list[layer_idx].ravel())[ind], row.ravel()[ind], col.ravel()[ind])).T
        # logger.info(f"table = {table[:5,:]}")
        logger.debug(f"shape table = {table.shape}")
        # table = np.vstack((edge_dict[layer_idx].ravel(), row.ravel(), col.ravel())).T
        # table = np.delete(table, np.where((np.asarray(list(map(itemgetter(0), table))) < threshold))[0], axis=0)
        # table = table[ np.asarray(list(map(itemgetter(0), table))) >= threshold, :]
        if layer_idx == 0:
            vec = list(zip(map(list, zip(map(lambda x: int(x), map(itemgetter(1), table)),
                                         map(lambda x: int(x), map(itemgetter(2), table)))),
                           map(itemgetter(0), table)))
        else:
            vec = vec + list(zip(map(list, zip(map(lambda x: int(x), map(itemgetter(1), table)),
                                               map(lambda x: int(x), map(itemgetter(2), table)))),
                                 map(itemgetter(0), table)))

    # Fast implementation
    # Adding the vertices
    nb_vertices = int(max([elem for array in tuple(map(itemgetter(0), vec)) for elem in array]))

    #logger.info(f"OK 2 !")

    dict_vertices = {key: [] for key in range(nb_vertices + 1)}
    for edge, timing in vec:
        if len(dict_vertices[edge[0]]) == 0 or timing <= min(dict_vertices[edge[0]]):
            dict_vertices[edge[0]].append(timing)
        if len(dict_vertices[edge[1]]) == 0 or timing <= min(dict_vertices[edge[1]]):
            dict_vertices[edge[1]].append(timing)
    for vertex in dict_vertices:
        if len(dict_vertices[vertex]) > 0:
            vec.append(([vertex], min(dict_vertices[vertex])))

    # Dionysus computations (persistent diagrams)
    #logger.info(f"Before filtration")
    f = Filtration()
    for vertices, timing in vec:
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
