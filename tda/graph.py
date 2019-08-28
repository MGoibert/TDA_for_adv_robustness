import networkx as nx
import typing
import logging
import torch
import numpy as np
from torch.nn import Module
from torch import Tensor
try:
    from torch_geometric.data import Data
except:
    Data = None
    logging.warning("torch_geometric wasn't found. You won't be able to use GNN algorithms."
                    "Please follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
                    " if you want to do so.")


class Graph(object):

    def __init__(self,
                 edge_dict: typing.Dict,
                 final_logits: typing.List[float],
                 original_data_point: typing.Optional[np.ndarray] = None
                 ):
        self._edge_dict = edge_dict
        self.original_data_point = original_data_point
        self.final_logits = final_logits

    @classmethod
    def from_model_and_data_point(cls,
                                  model: Module,
                                  x: Tensor,
                                  retain_data_point: bool = False):
        """
        Create graph from torch model and sample point
        """

        # Get the parameters of the model
        # Odd items are biases, so we just keep even elements (0, 2, etc.)
        w = list(model.parameters())[::2]

        # Get the neurons value for each layer (intermediate x)
        inter_x = model(x, return_intermediate=True)[1]

        # Get the edge weights
        # Step 1: compute the product of the weights and the intermediate x
        val = {}
        for k in range(len(w)):
            val[k] = (w[k] * inter_x[k]).detach().numpy()

        # Step 2: process (absolute value and rescaling)
        edge_dict = {key: 10e5 * np.abs(v) for key, v in val.items()}

        original_x = None
        if retain_data_point:
            original_x = x.detach().numpy()

        # Step 3: Saving the final logits
        final_logits = list(np.reshape(inter_x[-1].detach().numpy(), -1))

        return cls(
            edge_dict=edge_dict,
            final_logits=final_logits,
            original_data_point=original_x
        )

    def get_adjacency_matrix(
            self,
            threshold: typing.Optional[int] = None
    ) -> np.matrix:
        """
        Get the corresponding adjacency matrix
        """
        m0 = np.transpose(self._edge_dict[0])
        m1 = np.transpose(self._edge_dict[1])
        m2 = np.transpose(self._edge_dict[2])

        s0 = np.shape(m0)[0]
        s1 = np.shape(m1)[0]
        s2 = np.shape(m2)[0]
        s3 = np.shape(m2)[1]

        print((s0, s1, s2, s3))

        W = np.bmat([
            [np.zeros((s0, s0)), m0, np.zeros((s0, s2)), np.zeros((s0, s3))],
            [np.transpose(m0), np.zeros((s1, s1)), m1, np.zeros((s1, s3))],
            [np.zeros((s2, s0)), np.transpose(m1), np.zeros((s2, s2)), m2],
            [np.zeros((s3, s0)), np.zeros((s3, s1)), np.transpose(m2), np.zeros((s3, s3))],
        ])

        if threshold:
            W[W < threshold] = 0.0

        return W

    def get_layer_node_labels(self) -> typing.List[int]:
        """
        Return a list of label nodes equal to the layers they belong
        """
        m0 = np.transpose(self._edge_dict[0])
        m1 = np.transpose(self._edge_dict[1])
        m2 = np.transpose(self._edge_dict[2])

        s0 = np.shape(m0)[0]
        s1 = np.shape(m1)[0]
        s2 = np.shape(m2)[0]
        s3 = np.shape(m2)[1]

        return [0 for _ in range(s0)] + \
               [1 for _ in range(s1)] + \
               [2 for _ in range(s2)] + \
               [3 for _ in range(s3)]

    def to_nx_graph(
            self,
            threshold: typing.Optional[int] = None
    ) -> nx.Graph:
        return nx.from_numpy_matrix(self.get_adjacency_matrix(threshold))

    def to_pytorch_geometric_data(self, threshold: int) -> Data:
        offset = 0
        edge_indices = []
        edge_weights = []
        for key in self._edge_dict:
            rows, cols = np.where(self._edge_dict[key] >= threshold)
            edge_weights.append(torch.tensor(self._edge_dict[key][rows, cols]))
            cols = torch.tensor(cols, dtype=torch.long).unsqueeze(1) + offset
            offset += np.shape(self._edge_dict[key])[1]
            rows = torch.tensor(rows, dtype=torch.long).unsqueeze(1) + offset
            edge_indices.append(torch.cat((cols, rows), 1))

        edge_index = torch.transpose(torch.cat(edge_indices, 0), 0, 1)
        edge_weight = torch.cat(edge_weights, 0).unsqueeze(1)

        # Node labels as one-hot encoded version of the layer index
        x = torch.tensor(self.get_layer_node_labels(), dtype=torch.long).unsqueeze(1)
        x_onehot = torch.FloatTensor(len(x), len(self._edge_dict)+1)
        x_onehot.zero_()
        x_onehot.scatter_(1, x, 1)

        data = Data(
            x=x_onehot.type(torch.double),
            edge_index=edge_index,
            edge_attr=edge_weight)
        return data
