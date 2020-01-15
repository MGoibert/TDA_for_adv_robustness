import numpy as np
import pytest
import torch

from tda.graph import Graph
from tda.models.architectures import Architecture, LinearLayer, \
    ConvLayer, mnist_mlp, svhn_cnn_simple, svhn_lenet, SoftMaxLayer, svhn_resnet


def test_simple_graph():
    simple_archi: Architecture = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2),
            LinearLayer(2, 10),
            SoftMaxLayer()
        ])

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix().todense()

    from matplotlib import pyplot as plt
    plt.imshow(adjacency_matrix)
    plt.savefig("/Users/t.ricatte/test.png")

    assert np.shape(adjacency_matrix) == (19, 19)

    assert len(graph.get_edge_list()) == 38
    assert len(np.where(adjacency_matrix > 0)[0]) == 38 * 2

    print(graph.get_edge_list())
    print(simple_archi.get_pre_softmax_idx())


def test_simple_resnet_graph():
    simple_archi: Architecture = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 4),
            LinearLayer(4, 4),
            LinearLayer(4, 4),
            LinearLayer(4, 10),
            SoftMaxLayer()
        ],
        layer_links=[
            (-1, 0),
            (0, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 4)
        ]
    )

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix().todense()

    from matplotlib import pyplot as plt
    plt.imshow(adjacency_matrix)
    plt.savefig("/Users/t.ricatte/test.png")

    assert np.shape(adjacency_matrix) == (26, 26)

    assert len(graph.get_edge_list()) == 128
    assert len(np.where(adjacency_matrix > 0)[0]) == 128 * 2

    print(graph.get_edge_list())
    print(simple_archi.get_pre_softmax_idx())


def test_mnist_graph():

    simple_example = torch.randn((28, 28))

    graph = Graph.from_architecture_and_data_point(mnist_mlp, simple_example)

    adjacency_matrix = graph.get_adjacency_matrix().todense()

    assert np.shape(adjacency_matrix) == (1550, 1550)

    assert len(graph.get_edge_list()) == 522560
    assert len(np.where(adjacency_matrix > 0)[0]) == 522560 * 2

    print(graph.get_edge_list())


@pytest.mark.parametrize("stride,padding", [
    [1, 0],
    [2, 0],
    [1, 1],
    [2, 1],
    [3, 1]
])
def test_simple_cnn_one_channel(stride, padding):

    kernel_size = 2

    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            ConvLayer(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        ])

    simple_example = torch.tensor([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]]])

    nb_lines = 3
    nb_cols = 4

    for param in simple_archi.parameters():
        param.data = torch.tensor([[[
            [10, 20],
            [30, 40]
        ]]]).double()
        print(f"Kernel size is {list(param.shape)}")

    out = simple_archi(simple_example)
    expected_nb_lines = (nb_lines - kernel_size + 2 * padding) // stride + 1
    expected_nb_cols = (nb_cols - kernel_size + 2 * padding) // stride + 1
    assert np.shape(out) == (1, 1, expected_nb_lines, expected_nb_cols)

    m = simple_archi.get_graph_values(simple_example)

    print(m[(-1, 0)].todense())

    assert np.shape(m[(-1, 0)]) == (expected_nb_lines*expected_nb_cols, 12)


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
    assert np.shape(m[(-1, 0)]) == (18, 24)
    assert np.shape(m[(0, 1)]) == (1, 18)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (18+24+1, 18+24+1)


def test_svhn_graph():
    simple_example = torch.ones((3, 32, 32))*0.2

    for param in svhn_cnn_simple.parameters():
        param.data = torch.ones_like(param.data)*0.5

    graph = Graph.from_architecture_and_data_point(svhn_cnn_simple, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (11838, 11838)
    assert np.linalg.norm(adjacency_matrix.todense()) == 5798210602234079.0


#def test_svhn_resnet_graph():
#    simple_example = torch.randn((3, 32, 32))
#    out = svhn_resnet.forward(simple_example)
#    print(out)
#    print(out.shape)

#    graph = Graph.from_architecture_and_data_point(svhn_resnet, simple_example, use_sigmoid=False)
#    edge_list = graph.get_edge_list()
#    print(len(edge_list))


def test_svhn_lenet_graph():

    simple_example = torch.randn((3, 32, 32))
    graph = Graph.from_architecture_and_data_point(svhn_lenet, simple_example)

    assert len(graph._edge_dict) == svhn_lenet.get_nb_graph_layers()


if __name__ == "__main__":
    simple_example = torch.randn((3, 32, 32))
    out = svhn_resnet.forward(simple_example)
    print(out)
    print(out.shape)

    graph = Graph.from_architecture_and_data_point(svhn_resnet, simple_example)
    edge_list = graph.get_edge_list()
    print(len(edge_list))
