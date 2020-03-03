from torch import nn
from typing import Optional


class Layer(object):
    def __init__(self, func: nn.Module, graph_layer: bool, name: Optional[str] = None):
        self.func = func
        self.graph_layer = graph_layer
        self._activations = None
        self._matrix = None
        self.name = name

    def build_matrix(self):
        raise NotImplementedError()

    def get_matrix(self):
        raise NotImplementedError()

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))
