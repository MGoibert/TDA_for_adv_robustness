from typing import Optional

from scipy.sparse import coo_matrix
from torch import nn


class Layer(object):
    def __init__(self, func: nn.Module, graph_layer: bool, name: Optional[str] = None):
        self.func = func
        self.graph_layer = graph_layer
        self._activations = None
        self._matrix = None
        self.name = name

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()

    def get_matrix(self):
        ret = dict()
        for parentidx in self._activations:
            activ = self._activations[parentidx].reshape(-1)
            data_for_parent = [
                self._matrix.data[i] * float(activ[col_idx]) for i, col_idx in enumerate(self._matrix.col)
            ]
            ret[parentidx] = coo_matrix(
                (data_for_parent, (self._matrix.row, self._matrix.col)), self._matrix.shape
            )
        return ret

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))
