import logging
import typing

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from scipy.sparse import coo_matrix, bmat as sparse_bmat

from tda.models.architectures import Architecture

try:
    from torch_geometric.data import Data
except:
    Data = None
    logging.warning("torch_geometric wasn't found. You won't be able to use GNN algorithms."
                    "Please follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
                    " if you want to do so.")
logger = logging.getLogger(f"Graph")


class Graph(object):

    def __init__(self,
                 edge_dict: typing.Dict,
                 layer_links: typing.List,
                 final_logits: typing.List[float],
                 original_data_point: typing.Optional[np.ndarray] = None
                 ):
        self._edge_dict = edge_dict
        self._layer_links = layer_links

        self.original_data_point = original_data_point
        self.final_logits = final_logits

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

    @classmethod
    def from_architecture_and_data_point(cls,
                                         model: Architecture,
                                         x: Tensor,
                                         thresholds: typing.Optional[typing.Dict] = None,
                                         retain_data_point: bool = False,
                                         use_sigmoid: bool = True,
                                         dataset: str = "MNIST",
                                         architecture: str = "simple_fcn_mnist",
                                         epochs: int = 50
                                         ):
        raw_edge_dict = model.get_graph_values(x)

        edge_dict = dict()
        for layer_link in raw_edge_dict:
            v = raw_edge_dict[layer_link]
            v = np.abs(v) * 10e5
            if thresholds:
                # Keeping only edges below a given threhsold
                loc = v.data < thresholds[layer_link]
                v = coo_matrix((v.data[loc], (v.row[loc], v.col[loc])), np.shape(v))
                # Changing the sign for the persistent diagram
                v = -v
            if use_sigmoid:
                pass
                #logger.info(f"Using sigmoid for dataset {dataset}, archi {architecture} and epochs {epochs}")
                #file = f"stats/{dataset}_{architecture}_{str(epochs)}_epochs.npy"
                #v.data = np.where(v.data > 0, cls.use_sigmoid(
                #    data=v.data,
                #    layer_link=layer_link,
                #    file=file
                #), 0)

            edge_dict[layer_link] = v

        original_x = None
        if retain_data_point:
            original_x = x.detach().numpy()

        return cls(
            edge_dict=edge_dict,
            layer_links=model.layer_links,
            final_logits=list(),
            original_data_point=original_x
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
            target_idx, source_idx = mat.row, mat.col
            weights = mat.data
            source_idx += offset_source
            target_idx += offset_target

            for i in range(len(source_idx)):
                source_vertex = source_idx[i]
                target_vertex = target_idx[i]
                weight = weights[i]
                ret.append(([source_vertex, target_vertex], weight))
        return ret

    def get_adjacency_matrix(
            self
    ) -> np.matrix:
        """
        Get the corresponding adjacency matrix
        """

        shapes = self._get_shapes()

        print(shapes)
        all_layer_indices = sorted(list(shapes.keys()))

        bmat_list = tuple()
        for source_layer in all_layer_indices:
            bmat_row = tuple()
            for target_layer in all_layer_indices:
                if (source_layer, target_layer) in self._edge_dict:
                    # There is a connection between these layers
                    mat = np.transpose(self._edge_dict[(source_layer, target_layer)])
                elif (target_layer, source_layer) in self._edge_dict:
                    mat = self._edge_dict[(target_layer, source_layer)]
                    print(target_layer, source_layer)
                    print(np.shape(mat))
                else:
                    # There is no connection, let's create a zero matrix
                    # of the right shape !!
                    mat = coo_matrix((shapes[source_layer], shapes[target_layer]))
                bmat_row += (mat,)
            bmat_list += (bmat_row,)

        W = sparse_bmat(bmat_list, format="coo")

        return W

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

    # TODO: Make it DAG-ready
    def to_pytorch_geometric_data(self, threshold: int) -> Data:
        offset = 0
        edge_indices = []
        edge_weights = []
        for key, _ in enumerate(self._edge_list):
            rows, cols = np.where(self._edge_list[key] >= threshold)
            edge_weights.append(torch.tensor(self._edge_list[key][rows, cols]))
            cols = torch.tensor(cols, dtype=torch.long).unsqueeze(1) + offset
            offset += np.shape(self._edge_list[key])[1]
            rows = torch.tensor(rows, dtype=torch.long).unsqueeze(1) + offset
            edge_indices.append(torch.cat((cols, rows), 1))

        edge_index = torch.transpose(torch.cat(edge_indices, 0), 0, 1)
        edge_weight = torch.cat(edge_weights, 0).unsqueeze(1)

        # Node labels as one-hot encoded version of the layer index
        x = torch.tensor(self.get_layer_node_labels(), dtype=torch.long).unsqueeze(1)
        x_onehot = torch.FloatTensor(len(x), len(self._edge_list) + 1)
        x_onehot.zero_()
        x_onehot.scatter_(1, x, 1)

        data = Data(
            x=x_onehot.type(torch.double),
            edge_index=edge_index,
            edge_attr=edge_weight)
        return data
