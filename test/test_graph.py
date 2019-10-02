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


def test_simple_cnn_one_channel():
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

    for param in simple_archi.parameters():
        param.data = torch.tensor([[[
            [10, 20],
            [30, 40]
        ]]]).double()
        print(f"Kernel size is {list(param.shape)}")

    out = simple_archi(simple_example)
    print(out)

    m = simple_archi.get_graph_values(simple_example)

    print(m)

    assert np.shape(m[0]) == (6, 12)


def test_simple_cnn_multi_channels():
    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            # 2 input channels
            # 3 output channels
            ConvLayer(2, 3, 2)
        ])

    simple_example = torch.tensor([[
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ],
        [
            [2, 4, 6, 8],
            [10, 12, 14, 16],
            [18, 20, 22, 24]
        ]
    ]])

    for param in simple_archi.parameters():
        print(f"Kernel size is {list(param.shape)}")

    m = simple_archi.get_graph_values(simple_example)

    # Shape should be 6*3 out_channels = 18 x 12*2 in_channels = 24
    assert np.shape(m[0]) == (18, 24)


if __name__ == "__main__":
    test_simple_graph()
