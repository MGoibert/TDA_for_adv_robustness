import networkx as nx
import typing
import numpy as np
from torch.nn import Module
from torch import Tensor


class Graph(object):

    def __init__(self, edge_dict: typing.Dict):
        self._edge_dict = edge_dict

    @classmethod
    def from_model_and_data_point(cls,
                                  model: Module,
                                  x: Tensor):
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

        return cls(edge_dict=edge_dict)

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

        W = np.bmat([
            [np.zeros((s0, s0)), m0, np.zeros((s0, s2)), np.zeros((s0, s3))],
            [np.transpose(m0), np.zeros((s1, s1)), m1, np.zeros((s1, s3))],
            [np.zeros((s2, s0)), np.transpose(m1), np.zeros((s2, s2)), m2],
            [np.zeros((s3, s0)), np.zeros((s3, s1)), np.transpose(m2), np.zeros((s3, s3))],
        ])

        if threshold:
            W[W < threshold] = 0.0

        return W

    def to_nx_graph(
            self,
            threshold: typing.Optional[int] = None
    ) -> nx.Graph:
        return nx.from_numpy_matrix(self.get_adjacency_matrix(threshold))
