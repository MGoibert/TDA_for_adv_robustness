from typing import Optional

from scipy.sparse import coo_matrix, csr_matrix, diags
from torch import nn
import numpy as np

from tda.tda_logging import get_logger

logger = get_logger("Layer")
import torch
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)


class Layer(object):
    def __init__(self, func: nn.Module, graph_layer: bool, name: Optional[str] = None):
        self.func = func.type(default_tensor_type)
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
            ret[parentidx] = coo_matrix(
                self.matrix @ diags(activ.cpu().detach().numpy())
            )
        #  key = list(ret.keys())[0]
        #  a = {(r, c): d for r,c,d in zip(ret[key].row,ret[key].col,ret[key].data)}
        #  logger.info(f"{a}")
        return ret

    def get_matrix_thresholded(self, edges_to_keep_layer):

        # if len(edges_to_keep_layer)>0:
        #    row_to_keep = list(zip(*edges_to_keep_layer))[0]
        #    col_to_keep = list(zip(*edges_to_keep_layer))[1]
        #    m = csr_matrix(self.matrix)
        #    #m = self.matrix.todense()
        #    m[row_to_keep, col_to_keep] = 0.0
        # else:
        #    self.matrix = coo_matrix((np.shape(self.matrix)[0], np.shape(self.matrix)[1]))

        # This way is slower
        if len(edges_to_keep_layer) > 0:
            loc = [
                idx
                for idx, k in enumerate(zip(self.matrix.row, self.matrix.col))
                if k in edges_to_keep_layer
            ]
            self.matrix = coo_matrix(
                (self.matrix.data[loc], (self.matrix.row[loc], self.matrix.col[loc])),
                shape=np.shape(self.matrix),
            )
        else:
            # logger.info(f"In empty loop")
            self.matrix = coo_matrix(np.zeros(np.shape(self.matrix)))
        # logger.info(f"{self.matrix}")

    def process(self, x, store_for_graph):
        assert isinstance(x, dict)
        if store_for_graph:
            self._activations = x
        return self.func(sum(x.values()))

    def to(self, device):
        self.func.to(device)

    def __str__(self):
        return self.name or str(self.func)
