import numpy as np
import torch

from tda.embeddings.persistent_diagrams import compute_dgm_from_graph, \
    compute_dgm_from_graph, sliced_wasserstein_kernel
from tda.graph import Graph
from tda.models import Architecture
from tda.models.architectures import LinearLayer, SoftMaxLayer

from ripser import Rips
from persim.sliced_wasserstein import sliced_wasserstein


def test_ripser_vs_dionysus():
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

    k_dionysus = sliced_wasserstein_kernel(dgm1, dgm2)

    for pt in dgm1:
        print(pt)
    print(k_dionysus)

    ##########
    # Ripser #
    ##########

    ret = Rips(maxdim=2, coeff=1)
    #print(g1.get_adjacency_matrix())
    z = ret.fit_transform(g1.get_adjacency_matrix(), distance_matrix=True)

    Rips().plot(z)
    print(z)