import typing
import torch

import networkx as nx
import numpy as np
from torch import Tensor
from scipy.sparse import coo_matrix, bmat as sparse_bmat

from tda.models.architectures import Architecture
from tda.tda_logging import get_logger

logger = get_logger("GraphComputation")

import os
import pathlib
import pickle
import typing

import numpy as np
from operator import itemgetter


def use_sigmoid(v, layer_link, file, k="auto", quant=0.9):
    # dict_quant = torch.load("/Users/m.goibert/Documents/Criteo/P2_TDA_Detection/TDA_for_adv_robustness/cache/get_stats/architecture=svhn_lenet_dataset=svhn_dataset_size=100.cached")
    # data = data.todense()
    dict_quant = {
        (-1, 0): {0.5: 32476, 0.9: 147831, 0.99: 304033},
        (0, 1): {0.5: 792848, 0.9: 2538853, 0.99: 5852197},
        (1, 2): {0.5: 100556, 0.9: 580880, 0.99: 1958327},
        (2, 3): {0.5: 2549860, 0.9: 8806744, 0.99: 18652663},
        (3, 4): {0.5: 111805, 0.9: 656135, 0.99: 2082124},
        (4, 5): {0.5: 253505, 0.9: 1483325, 0.99: 4708494},
        (5, 6): {0.5: 1061315, 0.9: 7107406, 0.99: 22761988},
    }
    med = dict()
    qu = dict()
    for key_quant in dict_quant:
        if dict_quant[key_quant][0.5] != 1000000.0:
            med[key_quant] = dict_quant[key_quant][0.5]
            qu[key_quant] = dict_quant[key_quant][quant]
        else:
            med[key_quant] = 0.0
            qu[key_quant] = 1000000.0
    if k == "auto":
        k = -1 / (qu[layer_link] - med[layer_link]) * np.log(0.01 / 0.99)

    # val = 1/(1 + np.exp(-k * (np.asarray(data) - med[layer_link])))
    val = 1 / (1 + np.exp(-k * (v.data - med[layer_link])))
    return coo_matrix((val, (v.row, v.col)), np.shape(v))


class Graph(object):
    def __init__(self, edge_dict: typing.Dict):
        self._edge_dict = edge_dict

    @classmethod
    def from_architecture_and_data_point(cls, architecture: Architecture, x: Tensor):
        raw_edge_dict = architecture.get_graph_values(x)
        edge_dict = dict()
        for layer_link in raw_edge_dict:
            v = raw_edge_dict[layer_link]
            v = np.abs(v) * 10e5
            edge_dict[layer_link] = v

        return cls(edge_dict=edge_dict)

    def thresholdize(self, thresholds):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            # logger.info(f"layer link {layer_link} and shape of v = {v.todense().shape}")
            # Keeping only edges below a given threhsold
            loc = v.data < thresholds.get(layer_link, np.inf)
            v = coo_matrix(
                (v.data[loc].round(2), (v.row[loc], v.col[loc])), np.shape(v)
            )
            # Changing the sign for the persistent diagram
            self._edge_dict[layer_link] = v

    def thresholdize_underopt(self, ud):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            destination_layer = layer_link[1]
            if destination_layer in ud.keys():
                v2 = np.zeros(np.shape(v))
                loc = tuple(
                    [
                        list(map(itemgetter(0), ud[destination_layer])),
                        list(map(itemgetter(1), ud[destination_layer])),
                    ]
                )
                v2[loc] = v.todense()[loc]
                v = coo_matrix(v2)
            else:
                v = coo_matrix(np.zeros(np.shape(v)))
            self._edge_dict[layer_link] = v

    def sigmoidize(self, file=False, quant=0.99):
        dict_quant = {
            (-1, 0): {0.5: 32476, 0.9: 147831, 0.99: 304033},
            (0, 1): {0.5: 792848, 0.9: 2538853, 0.99: 5852197},
            (1, 2): {0.5: 100556, 0.9: 580880, 0.99: 1958327},
            (2, 3): {0.5: 2549860, 0.9: 8806744, 0.99: 18652663},
            (3, 4): {0.5: 111805, 0.9: 656135, 0.99: 2082124},
            (4, 5): {0.5: 253505, 0.9: 1483325, 0.99: 4708494},
            (5, 6): {0.5: 1061315, 0.9: 7107406, 0.99: 22761988},
        }
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            # Take median and "good" quantile to scale sigmoid
            med, qu = dict_quant[layer_link][0.5], dict_quant[layer_link][quant]
            k = -1 / (qu - med) * np.log(0.00001 / 0.99999)
            # Apply sigmoid
            val = 1 / (1 + np.exp(-k * (v.data - med)))
            v = coo_matrix((val, (v.row, v.col)), np.shape(v))
            self._edge_dict[layer_link] = v

    def _get_shapes(self):
        """
        Return the shape of the matrix for a given target layer
        (one layer can receive several matrices from its parents
        but they should all be of the same size)

        Therefore it's a mapping

        layer_idx -> shape

        This function is used as an helper to compute the list of edges
        and the adjacency matrix
        """
        shapes = {key[0]: np.shape(self._edge_dict[key])[1] for key in self._edge_dict}
        shapes.update(
            {key[1]: np.shape(self._edge_dict[key])[0] for key in self._edge_dict}
        )
        return shapes

    def get_edge_list(self):
        """
        Generate the list of edges of the multipartite graph
        """

        ret = list()

        shapes = self._get_shapes()
        all_layer_indices = sorted(list(shapes.keys()))
        vertex_offset = [0] + list(
            np.cumsum([shapes[idx] for idx in all_layer_indices])
        )
        vertex_offset = vertex_offset[:-1]

        for source_layer, target_layer in self._edge_dict:
            offset_source = vertex_offset[source_layer + 1]
            offset_target = vertex_offset[target_layer + 1]
            mat = self._edge_dict[(source_layer, target_layer)]

            for i in range(len(mat.data)):
                source_vertex = mat.col[i] + offset_source
                target_vertex = mat.row[i] + offset_target
                weight = mat.data[i]
                ret.append(([source_vertex, target_vertex], weight))
        return ret

    def get_adjacency_matrix(self) -> np.matrix:
        edges = self.get_edge_list()

        data = [e[1] for e in edges]
        row = [e[0][0] for e in edges]
        col = [e[0][1] for e in edges]

        # logger.info(f"Shape = {self._get_shapes().values()}")
        N = sum(self._get_shapes().values())
        mat = coo_matrix((data + data, (row + col, col + row)), shape=(N, N))

        return mat

    # TODO: Make it DAG-ready
    def get_layer_node_labels(self) -> typing.List[int]:
        """
        Return a list of label nodes equal to the layers they belong
        """

        n = len(self._edge_list)

        m = {key: np.transpose(self._edge_list[key]) for key in range(n)}
        s = {key: np.shape(m[key])[0] for key in range(n)}

        s[n] = np.shape(m[n - 1])[1]

        ret = list()
        for key in range(n + 1):
            ret += [key for _ in range(s[key])]

        return ret

    def to_nx_graph(self) -> nx.Graph:
        return nx.from_numpy_matrix(self.get_adjacency_matrix())
