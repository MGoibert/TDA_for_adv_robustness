import numpy as np
import torch

from tda.graph import Graph
from tda.models.architectures import Architecture, LinearLayer, \
    ConvLayer, mnist_mlp, svhn_cnn_simple, svhn_lenet


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

    from matplotlib import pyplot as plt
    plt.imshow(adjacency_matrix)
    plt.savefig("/Users/t.ricatte/test.png")

    assert np.shape(adjacency_matrix) == (19, 19)


def test_mnist_graph():

    simple_example = torch.randn((28, 28))

    graph = Graph.from_architecture_and_data_point(mnist_mlp, simple_example)

    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (1550, 1550)


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
            ConvLayer(2, 3, 2),
            LinearLayer(18, 1)
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
    assert np.shape(m[1]) == (1, 18)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (18+24+1, 18+24+1)


def test_svhn_graph(benchmark):

    def foo():
        simple_example = torch.randn((3, 32, 32))
        graph = Graph.from_architecture_and_data_point(svhn_cnn_simple, simple_example)
        adjacency_matrix = graph.get_adjacency_matrix()
        return np.shape(adjacency_matrix)

    assert benchmark(foo) == (11838, 11838)


def test_svhn_lenet_graph():

    simple_example = torch.randn((3, 32, 32))
    graph = Graph.from_architecture_and_data_point(svhn_lenet, simple_example)

    assert len(graph._edge_list) == svhn_lenet.get_nb_graph_layers()


if __name__ == "__main__":
    test_simple_graph()
