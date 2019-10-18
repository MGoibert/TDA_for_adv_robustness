#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import time
import typing
import numpy as np

from tda.graph import Graph
from tda.graph_dataset import get_dataset
from tda.models.architectures import mnist_mlp, get_architecture

#from igraph import Graph as IGraph
#from networkx.algorithms.centrality import betweenness_centrality, eigenvector_centrality
#from networkx.algorithms.centrality.katz import katz_centrality

start_time = time.time()

################
# Parsing args #
################

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=int, default=0)
parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=100)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphStats")

#####################
# Fetching datasets #
#####################

architecture = get_architecture(args.architecture)


def get_stats(epsilon: float, noise: float) -> typing.List:
    """
    Helper function to get list of embeddings
    """
    for line in get_dataset(
            num_epochs=args.epochs,
            epsilon=epsilon,
            noise=noise,
            adv=epsilon > 0.0,
            retain_data_point=False,
            architecture=architecture,
            source_dataset_name=args.dataset,
            dataset_size=args.dataset_size
        ):

        logger.info("Loading graph !")
        graph: Graph = line[0]

        for i, layer_matrix in enumerate(graph._edge_list):
            degrees = np.sum(layer_matrix, 1)

            q1_w = np.quantile(layer_matrix, 0.1)
            q9_w = np.quantile(layer_matrix, 0.9)

            q1_d = np.quantile(degrees, 0.1)
            q9_d = np.quantile(degrees, 0.9)

            logger.info(f"Weights layer {i}: {q1_w} {q9_w}")
            logger.info(f"Degrees layer {i}: {q1_d} {q9_d}")

        #my_igraph: IGraph = IGraph.Adjacency(graph.get_adjacency_matrix().tolist())

        #logger.info(f"Successfully created my igraph with {my_igraph.ecount()} edges !")


        #ec = my_igraph.eigenvector_centrality(directed=False)

        #logger.info(f"{len(ec)} {np.quantile(ec, 0.1)} {np.quantile(ec, 0.9)}")

        #nx_graph = graph.to_nx_graph()
        #logger.info("Successfully created nx graph !")

        #kc = katz_centrality(nx_graph)
        #logger.info(kc)

        #ec = eigenvector_centrality(nx_graph)
        #logger.info(ec)

        #bc = betweenness_centrality(nx_graph)
        #logger.info(bc)

        break

    return

get_stats(epsilon=0.0, noise=0.0)

end_time = time.time()

logger.info(f"Success in {end_time-start_time} seconds")
