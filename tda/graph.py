import logging
import typing

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from tda.models.architectures import Architecture

try:
    from torch_geometric.data import Data
except:
    Data = None
    logging.warning("torch_geometric wasn't found. You won't be able to use GNN algorithms."
                    "Please follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
                    " if you want to do so.")


class Graph(object):

    def __init__(self,
                 edge_list: typing.List,
                 final_logits: typing.List[float],
                 original_data_point: typing.Optional[np.ndarray] = None
                 ):
        self._edge_list = edge_list
        self.original_data_point = original_data_point
        self.final_logits = final_logits

    @classmethod
    def from_architecture_and_data_point(cls,
                                         model: Architecture,
                                         x: Tensor,
                                         thresholds: typing.Optional[typing.List[int]] = None,
                                         retain_data_point: bool = False
                                         ):
        raw_edge_list = model.get_graph_values(x)

        edge_list = list()
        for i, v in enumerate(raw_edge_list):
            v = np.abs(v) * 10e5
            if thresholds:
                v[v < thresholds[i]] = 0.0
            edge_list.append(v)

        original_x = None
        if retain_data_point:
            original_x = x.detach().numpy()

        return cls(
            edge_list=edge_list,
            final_logits=list(),
            original_data_point=original_x
        )

    def get_adjacency_matrix(
            self
    ) -> np.matrix:
        """
        Get the corresponding adjacency matrix
        """

        m = [np.transpose(x) for x in self._edge_list]

        s = [np.shape(x)[0] for x in m]
        n = len(self._edge_list)
        s.append(np.shape(m[n - 1])[1])

        bmat_list = tuple()
        for key_row in range(n + 1):
            bmat_row = tuple()
            for key_col in range(n + 1):
                if key_col == key_row + 1:
                    bmat_row += (m[key_row],)
                elif key_col == key_row - 1:
                    bmat_row += (np.transpose(m[key_col]),)
                else:
                    bmat_row += (np.zeros((s[key_row], s[key_col])),)
            bmat_list += (bmat_row,)

        W = np.bmat(bmat_list)

        return W

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
