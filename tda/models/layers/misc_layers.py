from scipy.sparse import coo_matrix

from .layer import Layer
from torch import nn
import torch
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)


class SoftMaxLayer(Layer):
    def __init__(self):
        super().__init__(func=nn.Softmax(dim=1), graph_layer=False)

    def get_matrix(self):
        raise NotImplementedError()

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()


class DropOut(Layer):
    def __init__(self):
        super().__init__(func=nn.Dropout(), graph_layer=False)

    def get_matrix(self):
        raise NotImplementedError()

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()


class ReluLayer(Layer):
    def __init__(self):
        super().__init__(func=nn.ReLU(), graph_layer=False)

    def process(self, x, store_for_graph):
        x = sum(x.values())
        out = self.func(x)
        return out

    def get_matrix(self):
        raise NotImplementedError()

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()


class BatchNorm2d(Layer):
    def __init__(self, channels, activ=None):
        self._activ = activ
        super().__init__(func=nn.BatchNorm2d(num_features=channels), graph_layer=False)
        self._activ = activ

    def process(self, x, store_for_graph):
        x = sum(x.values())
        out = self.func(x)
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out

    def get_matrix(self):
        raise NotImplementedError()

    def build_matrix(self) -> coo_matrix:
        raise NotImplementedError()
