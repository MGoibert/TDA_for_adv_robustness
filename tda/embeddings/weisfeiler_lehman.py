import networkx as nx
import typing
import scipy
from collections import OrderedDict


def _count_patterns(d: typing.Dict, l: typing.List):
    for pat in l:
        d[pat] = d.get(pat, 0) + 1


def _get_hash_value(s: str, max_val: int):
    return hash(f"n{s}") % max_val


def get_wl_embedding(
        graph: nx.Graph,
        height: int = 1,
        hash_size: int = 100,
        input_labels: typing.Optional[typing.List] = None):
    counters = OrderedDict()
    labels = input_labels if input_labels is not None else [1 for _ in graph.nodes()]
    labels = [_get_hash_value(str(l), hash_size) for l in labels]
    _count_patterns(counters, labels)

    for it in range(height):
        # print(f"Starting iteration {it}...")
        new_labels = list()
        for node in graph.nodes():
            neighbors = graph.neighbors(node)
            labels_neighbors = ''.join(sorted([str(labels[x]) for x in neighbors]))
            new_labels.append(f"{labels[node]},{labels_neighbors}")
        labels = [_get_hash_value(str(l), hash_size) for l in new_labels]
        _count_patterns(counters, labels)

    # print(counters)
    data = list(counters.values())
    row = list(counters.keys())
    col = [0 for _ in row]
    emb = scipy.sparse.coo_matrix((data, (row, col)), shape=(hash_size, 1))
    return emb
