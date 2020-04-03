from typing import Optional

from scipy.sparse import coo_matrix, diags
from torch import nn


class Layer(object):
    def __init__(self, func: nn.Module, graph_layer: bool, name: Optional[str] = None):
        self.func = func
        self.graph_layer = graph_layer
        self._activations = None
        self.matrix = None
        self.name = name

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()

    def get_matrix(self):
        ret = dict()
        for parentidx in self._activations:
            activ = self._activations[parentidx].reshape(-1)
            ret[parentidx] = coo_matrix(self.matrix @ diags(activ.detach().numpy()))
        return ret

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))
