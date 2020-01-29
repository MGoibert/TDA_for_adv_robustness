import typing

import networkx as nx
import numpy as np
from torch import Tensor
from scipy.sparse import coo_matrix, bmat as sparse_bmat

from tda.models.architectures import Architecture
from tda.logging import get_logger

logger =  get_logger("Graph")


class Graph(object):

    def __init__(self,
                 edge_dict: typing.Dict
                 ):
        self._edge_dict = edge_dict

    @staticmethod
    def use_sigmoid(data, layer_link, file, k="auto", quant=0.9):
        dict_quant = np.load(file, allow_pickle=True).flat[0]
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

        val = 1/(1 + np.exp(-k * (np.asarray(data) - med[layer_link])))
        return list(val)

    #@classmethod
    #def from_architecture_and_data_point_raw_dict(cls,
    #                                     architecture: Architecture,
    #                                     x: Tensor
    #                                     ):
    #    raw_edge_dict = architecture.get_graph_values(x)
    #    layer_links = list()

    #    edge_dict = dict()
    #    for layer_link in raw_edge_dict:
    #        layer_links.append(layer_link)
    #        v = raw_edge_dict[layer_link]
    #        v = np.abs(v) * 10e5
    #        edge_dict[layer_link] = v
    #    #logger.info(f"check edge dict raw = {edge_dict[(5,6)].todense().sum()}")
    #    return edge_dict, layer_links
        

    #@classmethod
    #def from_architecture_and_data_point(cls,
    #                                     edge_dict: typing.Dict,
    #                                     layer_links: typing.List,
    #                                     thresholds: typing.Optional[typing.Dict] = None
    #                                     ):
    #    for layer_link in layer_links:
    #        if thresholds:
    #            # Keeping only edges below a given threhsold
    #            loc = edge_dict[layer_link].data < thresholds[layer_link]
    #            edge_dict[layer_link] = coo_matrix((edge_dict[layer_link].data[loc],
    #                                    (edge_dict[layer_link].row[loc],
    #                                    edge_dict[layer_link].col[loc])),
    #                                        np.shape(edge_dict[layer_link]))
    #            # Changing the sign for the persistent diagram
    #            edge_dict[layer_link] = -edge_dict[layer_link]
    #    #logger.info(f"check edge dict = {edge_dict[(0,1)].todense().sum()}")

    #    return cls(
    #        edge_dict=edge_dict,
    #        layer_links=layer_links,
    #        final_logits=list()
    #    )

    @classmethod
    def from_architecture_and_data_point(cls,
                                         architecture: Architecture,
                                         x: Tensor,
                                         thresholds: typing.Optional[typing.Dict] = None
                                         ):
        raw_edge_dict = architecture.get_graph_values(x)
        edge_dict = dict()
        for layer_link in raw_edge_dict:
            v = raw_edge_dict[layer_link]
            v = np.abs(v) * 10e5
            if thresholds is not None:
                # Keeping only edges below a given threhsold
                loc = v.data < thresholds.get(layer_link, np.inf)
                v = coo_matrix((v.data[loc], (v.row[loc], v.col[loc])), np.shape(v))
                # Changing the sign for the persistent diagram
                v = -v

            edge_dict[layer_link] = v

        return cls(
            edge_dict=edge_dict
        )

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
        shapes.update({key[1]: np.shape(self._edge_dict[key])[0] for key in self._edge_dict})
        return shapes

    def get_edge_list(self):
        """
        Generate the list of edges of the multipartite graph
        """

        ret = list()

        shapes = self._get_shapes()
        all_layer_indices = sorted(list(shapes.keys()))
        vertex_offset = [0] + list(np.cumsum([
            shapes[idx]
            for idx in all_layer_indices
        ]))
        vertex_offset = vertex_offset[:-1]

        for source_layer, target_layer in self._edge_dict:
            offset_source = vertex_offset[source_layer+1]
            offset_target = vertex_offset[target_layer+1]
            mat = self._edge_dict[(source_layer, target_layer)]

            for i in range(len(mat.data)):
                source_vertex = mat.col[i] + offset_source
                target_vertex = mat.row[i] + offset_target
                weight = mat.data[i]
                ret.append(([source_vertex, target_vertex], weight))
        return ret

    def get_adjacency_matrix(
            self
    ) -> np.matrix:
        edges = self.get_edge_list()

        data = [e[1] for e in edges]
        row = [e[0][0] for e in edges]
        col = [e[0][1] for e in edges]

        N = sum(self._get_shapes().values())
        mat = coo_matrix((data, (row, col)), shape=(N, N))

        return mat

    # TODO: Make it DAG-ready
    def get_layer_node_labels(self) -> typing.List[int]:
        """
        Return a list of label nodes equal to the layers they belong
        """

        n = len(self._edge_list)

        m = {
            key: np.transpose(self._edge_list[key])
            for key in range(n)
        }
        s = {
            key: np.shape(m[key])[0]
            for key in range(n)
        }

        s[n] = np.shape(m[n - 1])[1]

        ret = list()
        for key in range(n + 1):
            ret += [key for _ in range(s[key])]

        return ret

    def to_nx_graph(
            self
    ) -> nx.Graph:
        return nx.from_numpy_matrix(self.get_adjacency_matrix())
