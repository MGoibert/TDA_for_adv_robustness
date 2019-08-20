import typing
import numpy as np
from operator import itemgetter
try:
    from dionysus import Filtration, Simplex, homology_persistence, init_diagrams
except Exception as e:
    print("Unable to find dionysus")
    Filtration = None
    Simplex = None
    homology_persistence = None
    init_diagrams = None


def compute_dgm_from_edges(
        edge_dict: typing.Dict,
        threshold: int
):
    """
    Create the simplicial complexes using dionysus
    Fast implementation but "by hand"
    """

    vec = []
    shape = np.cumsum([edge_dict[key].shape[1] for key in edge_dict.keys()])
    shape = np.insert(shape, 0, 0)
    shape = np.insert(shape, len(shape), shape[len(shape) - 1] + edge_dict[len(shape) - 2].shape[0])

    for layer_idx in edge_dict.keys():
        # Adding the edges
        row, col = np.meshgrid(np.arange(shape[layer_idx], shape[layer_idx + 1]),
                               np.arange(shape[layer_idx + 1], shape[layer_idx + 2]))
        table = np.vstack((edge_dict[layer_idx].ravel(), row.ravel(), col.ravel())).T
        table = np.delete(table, np.where((np.asarray(list(map(itemgetter(0), table))) < threshold))[0], axis=0)
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
    f = Filtration()
    for vertices, timing in vec:
        f.append(Simplex(vertices, timing))
    f.sort()
    m = homology_persistence(f)
    dgms = init_diagrams(m, f)

    return dgms[0]
