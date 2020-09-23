from scipy.sparse import coo_matrix
from torch import Tensor
import typing
import numpy as np
from tda.models.architectures import Architecture
from tda.tda_logging import get_logger

logger = get_logger("GraphComputation")


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

    def thresholdize(self, thresholds, low_pass: bool):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            # logger.info(f"layer link {layer_link} and shape of v = {v.todense().shape}")
            # Keeping only edges below a given threhsold
            if low_pass:
                loc = v.data < thresholds.get(layer_link, np.inf)
            else:
                loc = v.data > thresholds.get(layer_link, np.inf)
            v = coo_matrix(
                (v.data[loc].round(2), (v.row[loc], v.col[loc])), np.shape(v)
            )
            # Changing the sign for the persistent diagram
            self._edge_dict[layer_link] = v

    def thresholdize_underopt(self, ud):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            if layer_link[1] in ud.keys():
                loc = [
                    idx
                    for idx, k in enumerate(zip(v.row, v.col))
                    if k in ud[layer_link[1]]
                ]
                self._edge_dict[layer_link] = coo_matrix(
                    (v.data[loc], (v.row[loc], v.col[loc])), shape=np.shape(v)
                )
            else:
                self._edge_dict[layer_link] = coo_matrix(np.zeros(np.shape(v)))

    def thresholdize_per_graph(self, thresholds: typing.Dict, low_pass: bool):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            q = np.quantile(v.data, thresholds.get(layer_link, 0.0))
            if low_pass:
                loc = v.data < q
            else:
                loc = v.data > q
            v = coo_matrix((v.data[loc], (v.row[loc], v.col[loc])), np.shape(v))
            # Changing the sign for the persistent diagram
            self._edge_dict[layer_link] = v

    # def thresholdize_underopt(self, ud):
    #    for layer_link in self._edge_dict:
    #        v = self._edge_dict[layer_link]
    #        destination_layer = layer_link[1]
    #        if destination_layer in ud.keys():
    #            v2 = np.zeros(np.shape(v))
    #            loc = tuple(
    #                [
    #                    list(map(itemgetter(0), ud[destination_layer])),
    #                    list(map(itemgetter(1), ud[destination_layer])),
    #                ]
    #            )
    #            v2[loc] = v.todense()[loc]
    #            v = coo_matrix(v2)
    #        else:
    #            v = coo_matrix(np.zeros(np.shape(v)))
    #        self._edge_dict[layer_link] = v

    def sigmoidize(self, quantiles_helpers, quant=0.999):
        for layer_link in self._edge_dict:
            v = self._edge_dict[layer_link]
            # Take median and "good" quantile to scale sigmoid
            qs = quantiles_helpers[layer_link].get_quantiles([0.5, quant])
            med = qs[0]
            qu = qs[1]

            k = -1 / (qu - med) * np.log(0.001 / 0.999)
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
            offset_source = vertex_offset[all_layer_indices.index(source_layer)]
            offset_target = vertex_offset[all_layer_indices.index(target_layer)]
            mat = self._edge_dict[(source_layer, target_layer)]

            for i in range(len(mat.data)):
                source_vertex = mat.col[i] + offset_source
                target_vertex = mat.row[i] + offset_target
                weight = mat.data[i]
                ret.append(([source_vertex, target_vertex], weight))
        return ret

    def get_adjacency_matrix(self) -> coo_matrix:
        edges = self.get_edge_list()

        data = [e[1] for e in edges]
        row = [e[0][0] for e in edges]
        col = [e[0][1] for e in edges]

        # logger.info(f"Shape = {self._get_shapes().values()}")
        N = sum(self._get_shapes().values())
        mat = coo_matrix((data + data, (row + col, col + row)), shape=(N, N))

        return mat
