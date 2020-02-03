import numpy as np
import torch
from ripser import Rips

from tda.embeddings.persistent_diagrams import (compute_dgm_from_graph_ripser,
                                                compute_dgm_from_graph_ripser as compute_dgm_from_graph,
                                                sliced_wasserstein_kernel)
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import LinearLayer, SoftMaxLayer

simple_archi: Architecture = Architecture(
        preprocess=lambda x: x,
        layers=[
            LinearLayer(4, 3),
            LinearLayer(3, 2),
            LinearLayer(2, 10),
            SoftMaxLayer()
        ])

ex1 = torch.ones(4) * 42
ex2 = torch.ones(4) * 37

g1 = Graph.from_architecture_and_data_point(simple_archi, ex1, thresholds=dict())
g2 = Graph.from_architecture_and_data_point(simple_archi, ex2, thresholds=dict())

############
# Dionysus #
############

dgm1 = compute_dgm_from_graph(g1)
dgm2 = compute_dgm_from_graph(g2)


def test_sliced_wassertstein_kernel(benchmark):

    def compute_dgm():
        return sliced_wasserstein_kernel(dgm1, dgm2)

    k_dionysus = benchmark(compute_dgm)
    assert np.isclose(k_dionysus, 7.450580596923829e-10, rtol=1e-12)

    rips = Rips(maxdim=1, coeff=2)

    dgm1_ripser = compute_dgm_from_graph_ripser(g1)

    print("Dionysus")
    print(dgm1)
    for pt in dgm1:
        print(pt)

    print("----")
    print("Ripser")
    print(dgm1_ripser)




