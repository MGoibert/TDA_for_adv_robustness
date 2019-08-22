import typing
import scipy
from collections import OrderedDict
from tda.graph import Graph


class NodeLabels(object):
    # Label all nodes with 1
    NONE = "none"
    # Use the index of the layer as the initial node label
    LAYERS = "layers"


def _count_patterns(d: typing.Dict, l: typing.List):
    for pat in l:
        d[pat] = d.get(pat, 0) + 1


def _get_hash_value(s: str, max_val: int):
    return hash(f"n{s}") % max_val


def get_wl_embedding(
        graph: Graph,
        threshold: int = 0,
        height: int = 1,
        hash_size: int = 100,
        node_labels: str = NodeLabels.NONE):
    counters = OrderedDict()
    nxgraph = graph.to_nx_graph(threshold=threshold)

    if node_labels == NodeLabels.LAYERS:
        labels = graph.get_layer_node_labels()
    elif node_labels == NodeLabels.NONE:
        labels = [1 for _ in nxgraph.nodes()]
    else:
        raise NotImplementedError(f"Unknown labels {node_labels}")

    labels = [_get_hash_value(str(l), hash_size) for l in labels]
    _count_patterns(counters, labels)

    for it in range(height):
        # print(f"Starting iteration {it}...")
        new_labels = list()
        for node in nxgraph.nodes():
            neighbors = nxgraph.neighbors(node)
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
