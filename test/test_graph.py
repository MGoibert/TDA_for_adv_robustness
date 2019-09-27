import numpy as np
import torch

from tda.graph import Graph
from tda.models.architectures import Architecture, LinearLayer, ConvLayer


def test_simple_graph():
    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2),
            LinearLayer(2, 10)
        ])

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (9, 9)


def test_simple_cnn():
    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            ConvLayer(1, 1, 2)
        ])

    simple_example = torch.tensor([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]]])

    print(simple_example)

    for param in simple_archi.parameters():
        param.data = torch.tensor([[[
            [10, 20],
            [30, 40]
        ]]]).double()

    out = simple_archi(simple_example)
    print(out)

    M = simple_archi.get_graph_values(simple_example)

    print(M)

if __name__ == "__main__":
    test_simple_graph()
