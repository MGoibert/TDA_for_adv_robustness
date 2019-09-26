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

    logging.warn(simple_archi)

    simple_example = torch.ones(4)

    out = simple_archi(simple_example)

    print(simple_archi.get_graph_values(simple_example))


if __name__ == "__main__":
    test_simple_graph()
