from torch import nn
from .layer import Layer
import numpy as np
from scipy.sparse import coo_matrix


class MaxPool2dLayer(Layer):
    def __init__(self, kernel_size, stride=None, activ=None):

        if stride is None:
            stride = kernel_size

        super().__init__(
            func=nn.MaxPool2d(kernel_size, stride, return_indices=True), graph_layer=True
        )

        self._activ = activ
        # self._use_activation = True

    def build_matrix(self) -> coo_matrix:
        # Unfortunately, we cannot precompute the matrix
        # for MaxPool2dLayers
        self.matrix = None

    def get_matrix(self):
        """
        Return the weight of the linear layer, ignore biases
        """
        idx = self._indx
        for i in range(idx.shape[1]):
            idx[:, i, :, :] = (
                idx[:, i, :, :]
                + i * self._activations_shape[2] * self._activations_shape[3]
            )
        idx = idx.cpu().numpy().flatten()
        dim = 1
        dim_out = 1
        for d in self._activations_shape:
            dim *= d
        for d in self._out_shape:
            dim_out *= d
        # print("dim =", dim, "and dim output =", dim_out)
        #m = np.zeros((dim, dim_out))
        #for i in range(dim_out):
        #    if True:  # self._use_activation:
        #        m[:, i][idx[i]] = self._out.flatten(0)[
        #            i
        #        ]  # self._activations.flatten(0)[idx[i]]
        #    else:
        #        m[:, i][idx[i]] = 1
        #return {
        #    parentidx: coo_matrix(np.matrix(m.transpose()))
        #    for parentidx in self._parent_indices
        #}
        return {
            parentidx: coo_matrix(np.zeros((dim, dim_out)))
            for parentidx in self._parent_indices
        }

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        x_sum = sum(x.values()).double()
        out, indx = self.func(x_sum)
        if store_for_graph:
            self._activations = x
            self._indx = indx
            self._out_shape = out.shape
            self._activations_shape = x_sum.shape
            self._activations_values = x_sum
        if self._activ:
            if type(self._activ) == list:
                for act in self._activ:
                    out = act(out)
                return out
            else:
                return self._activ(out)
        else:
            return out
