from torch import nn
from .layer import Layer
from scipy.sparse import coo_matrix
import numpy as np


class LinearLayer(Layer):
    def __init__(self, in_width, out_width, activ=None, name=None):

        super().__init__(
            func=nn.Linear(in_width, out_width), graph_layer=True, name=name
        )

        self._in_width = in_width
        self._activ = activ

    def build_matrix(self) -> coo_matrix:
        matrix = list(self.func.parameters())[0]
        self._matrix = coo_matrix(matrix.cpu().detach().numpy())
        return self._matrix

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
        _x = {key: x[key].reshape(-1, self._in_width) for key in x}
        if store_for_graph:
            self._activations = _x
        _x = sum(_x.values())
        if self._activ:
            out = self.func(_x)
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return self.func(_x)
