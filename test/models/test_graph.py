import numpy as np
import pytest
import torch

from tda.embeddings import get_embedding, EmbeddingType, KernelType
from tda.embeddings import get_gram_matrix
from tda.graph import Graph
from tda.models import get_deep_model, Dataset
from tda.models.architectures import (
    Architecture,
    svhn_cnn_simple,
    svhn_lenet,
)
from tda.models.architectures import mnist_mlp, get_architecture
from tda.models.layers import LinearLayer, ConvLayer, SoftMaxLayer
from tda.protocol import get_protocolar_datasets
from tda.thresholds import process_thresholds


def test_simple_graph():
    simple_archi: Architecture = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2),
            LinearLayer(2, 10),
            SoftMaxLayer(),
        ],
    )
    simple_archi.build_matrices()

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix().todense()

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
            SoftMaxLayer(),
        ],
        layer_links=[(-1, 0), (0, 1), (1, 2), (1, 3), (2, 3), (3, 4)],
    )
    simple_archi.build_matrices()

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix().todense()

    assert np.shape(adjacency_matrix) == (26, 26)

    assert len(graph.get_edge_list()) == 128
    assert len(np.where(adjacency_matrix > 0)[0]) == 128 * 2

    print(graph.get_edge_list())
    print(simple_archi.get_pre_softmax_idx())


def test_mnist_graph():

    simple_example = torch.randn((28, 28))

    mnist_mlp.build_matrices()
    graph = Graph.from_architecture_and_data_point(mnist_mlp, simple_example)

    adjacency_matrix = graph.get_adjacency_matrix().todense()

    assert np.shape(adjacency_matrix) == (1550, 1550)

    assert len(graph.get_edge_list()) == 522560
    assert len(np.where(adjacency_matrix > 0)[0]) == 522560 * 2

    print(graph.get_edge_list())


@pytest.mark.parametrize("stride,padding", [[1, 0], [2, 0], [1, 1], [2, 1], [3, 1]])
def test_simple_cnn_one_channel(stride, padding):

    kernel_size = 2

    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            ConvLayer(
                in_channels=1,
                out_channels=1,
                input_shape=(3, 4),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ],
    )
    simple_archi.build_matrices()

    simple_example = torch.tensor([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]])

    nb_lines = 3
    nb_cols = 4

    for param in simple_archi.parameters():
        param.data = torch.tensor([[[[10, 20], [30, 40]]]]).double()
        print(f"Kernel size is {list(param.shape)}")

    out = simple_archi(simple_example)
    expected_nb_lines = (nb_lines - kernel_size + 2 * padding) // stride + 1
    expected_nb_cols = (nb_cols - kernel_size + 2 * padding) // stride + 1
    assert np.shape(out) == (1, 1, expected_nb_lines, expected_nb_cols)

    m = simple_archi.get_graph_values(simple_example)

    print(m[(-1, 0)].todense())

    assert np.shape(m[(-1, 0)]) == (expected_nb_lines * expected_nb_cols, 12)


def test_simple_cnn_multi_channels():
    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            # 2 input channels
            # 3 output channels
            ConvLayer(2, 3, 2, input_shape=(3, 4)),
            LinearLayer(18, 1),
        ],
    )
    simple_archi.build_matrices()

    simple_example = torch.tensor(
        [
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[2, 4, 6, 8], [10, 12, 14, 16], [18, 20, 22, 24]],
            ]
        ]
    )

    for param in simple_archi.parameters():
        print(f"Kernel size is {list(param.shape)}")

    m = simple_archi.get_graph_values(simple_example)

    # Shape should be 6*3 out_channels = 18 x 12*2 in_channels = 24
    assert np.shape(m[(-1, 0)]) == (18, 24)
    assert np.shape(m[(0, 1)]) == (1, 18)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (18 + 24 + 1, 18 + 24 + 1)


def test_svhn_graph():
    simple_example = torch.ones((3, 32, 32)) * 0.2

    for param in svhn_cnn_simple.parameters():
        param.data = torch.ones_like(param.data) * 0.5

    svhn_cnn_simple.forward(simple_example)
    svhn_cnn_simple.build_matrices()

    graph = Graph.from_architecture_and_data_point(svhn_cnn_simple, simple_example)
    adjacency_matrix = graph.get_adjacency_matrix()

    assert np.shape(adjacency_matrix) == (11838, 11838)
    assert np.linalg.norm(adjacency_matrix.todense()) == 5798210602234079.0


# def test_svhn_resnet_graph():
#    simple_example = torch.randn((3, 32, 32))
#    out = svhn_resnet.forward(simple_example)
#    print(out)
#    print(out.shape)

#    graph = Graph.from_architecture_and_data_point(svhn_resnet, simple_example, use_sigmoid=False)
#    edge_list = graph.get_edge_list()
#    print(len(edge_list))


def test_svhn_lenet_graph():

    simple_example = torch.randn((3, 32, 32))

    svhn_lenet.forward(simple_example)
    svhn_lenet.build_matrices()

    graph = Graph.from_architecture_and_data_point(svhn_lenet, simple_example)

    assert len(graph._edge_dict) == svhn_lenet.get_nb_graph_layers()


def previous_kernel_version(DGM1, DGM2, param_space):
    def sliced_wasserstein_kernel(dgm1, dgm2, param_space):
        # logger.info(f"Sliced Wass. Kernel ")
        M = param_space[0]["M"]
        vec1 = []
        vec2 = []
        birth1 = list()
        birth2 = list()
        for i, pt1 in enumerate(dgm1):
            if pt1[1] == 18446744073709551616:
                birth1.append(pt1[0])
            vec1.append([pt1[0], pt1[1]])
            vec2.append([(pt1[0] + pt1[1]) / 2.0, (pt1[0] + pt1[1]) / 2.0])
        for i, pt2 in enumerate(dgm2):
            if pt2[1] == 18446744073709551616:
                birth2.append(pt2[0])
            vec2.append([pt2[0], pt2[1]])
            vec1.append([(pt2[0] + pt2[1]) / 2.0, (pt2[0] + pt2[1]) / 2.0])
        # print(f"Births of inf? dgm1 = {birth1} and dgm2 = {birth2}")
        sw = 0
        theta = -np.pi / 2
        s = np.pi / M
        for i in range(M):
            v1 = [np.dot(pt1, [theta, theta]) for pt1 in vec1]
            v2 = [np.dot(pt2, [theta, theta]) for pt2 in vec2]
            li1 = sorted(zip(v1, vec1))
            li2 = sorted(zip(v2, vec2))
            v1, vec1 = map(list, zip(*li1))
            v2, vec2 = map(list, zip(*li2))
            # v1.sort()
            # v2.sort()
            # print(f"Vec 1 around 0 : {vec1[0:3]} and vec 2 : {vec2[0:3]}")
            val = np.nan_to_num(np.asarray(v1) - np.asarray(v2))
            # for idx, elem in enumerate(val):
            #    if abs(elem) > 1e-10:
            #        print(f"Step M = {i} and theta = {theta}")
            #        print(f"elem > 0  for idx {idx} --> val: {val[idx]}, v1: {np.asarray(v1)[idx]} and vec1: {vec1[idx]}")
            #        print(f"Same for 2 v2: {np.asarray(v2)[idx]} and vec2: {vec2[idx]} \n")
            sw = sw + s * np.linalg.norm(val, ord=1)
            theta = theta + s
            # logger.info(f"End Sliced Wass. Kernel")
            # print("Run :", i, " and sw =", (1/np.pi)*sw)
        return (1 / np.pi) * sw

    sigma = param_space[0]["sigma"]
    n = len(DGM1)
    m = len(DGM2)
    gram = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            # print(f"Row {i} and col {j}")
            sw = sliced_wasserstein_kernel(DGM1[i], DGM2[j], param_space=param_space)
            gram[i, j] = np.exp(-sw / (2 * sigma ** 2))
    return gram


def test_kernel():

    return

    architecture = get_architecture("svhn_lenet")
    dataset = Dataset.get_or_create("SVHN")
    architecture = get_deep_model(
        num_epochs=200, dataset=dataset, architecture=architecture, train_noise=0.0
    )

    thresholds = process_thresholds(
        raw_thresholds="0.4_0.05_0.4_0.05_0_0_0",
        dataset=dataset,
        architecture=architecture,
        dataset_size=4,
    )

    all_epsilons = [0.05]
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=0.0,
        dataset=dataset,
        succ_adv=False,
        archi=architecture,
        dataset_size=4,
        attack_type="FGSM",
        all_epsilons=all_epsilons,
    )
    print(f"Here 2 train clean {len(train_clean)} and adv = {len(test_adv[0.05])}")
    clean = list()
    adv = list()
    for line in train_clean:
        clean.append(
            get_embedding(
                embedding_type=EmbeddingType.PersistentDiagram,
                line=line,
                params={"hash_size": 0, "height": 0, "node_labels": 0, "steps": 0},
                architecture=architecture,
                thresholds=thresholds,
            )
        )
    for line in test_adv[0.05]:
        adv.append(
            get_embedding(
                embedding_type=EmbeddingType.PersistentDiagram,
                line=line,
                params={"hash_size": 0, "height": 0, "node_labels": 0, "steps": 0},
                architecture=architecture,
                thresholds=thresholds,
            )
        )
    print(f"Here 3 clean = {len(clean)} and adv = {len(adv)}")
    param_space = [{"M": 20, "sigma": 0.5}]
    print(f"Clean matrix ! \n")
    gram_train_matrices = get_gram_matrix(
        kernel_type=KernelType.SlicedWasserstein,
        embeddings_in=clean,
        embeddings_out=None,
        params=param_space,
    )
    print(f"Adv matrix ! \n")
    gram_adv_matrices = get_gram_matrix(
        kernel_type=KernelType.SlicedWasserstein,
        embeddings_in=adv,
        embeddings_out=None,
        params=param_space,
    )
    print(f"Clean vs adv matrix ! \n")
    gram_cleanvsadv_matrices = get_gram_matrix(
        kernel_type=KernelType.SlicedWasserstein,
        embeddings_in=list(clean) + list(adv),
        embeddings_out=adv,
        params=param_space,
    )
    print(f"gram mat clean = {gram_train_matrices}")
    print(f"gram mat adv = {gram_adv_matrices}")
    print(f"gram mat clean vs adv = {gram_cleanvsadv_matrices}")

    # print(f"Previous version: clean !!")
    # gram2 = previous_kernel_version(clean, clean, param_space)
    # print(f"Previous version: adv !!")
    # gram2_adv = previous_kernel_version(adv, adv, param_space)
    # rint(f"Previous version: clean vs adv !!")
    # gram2_cleanvsadv = previous_kernel_version(clean, adv, param_space)
    # print(f"gram2 clean mat = {gram2}")
    # print(f"gram2 adv mat = {gram2_adv}")
    # print(f"gram2 clean vs adv mat = {gram2_cleanvsadv}")


if __name__ == "__main__":
    test_kernel()
