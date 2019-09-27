from collections import OrderedDict
import torch
import logging

from tda.models.architectures import Architecture, LinearLayer
from tda.graph import Graph


def test_simple_graph():
    simple_archi = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2)
        ])

    simple_example = torch.ones(4)

    graph = Graph.from_architecture_and_data_point(simple_archi, simple_example)

    print(graph._edge_dict)

    print(graph.get_adjacency_matrix())

if __name__ == "__main__":
    test_simple_graph()
