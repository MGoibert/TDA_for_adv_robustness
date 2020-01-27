from typing import List, Optional, Dict
import numpy as np
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import sliced_wasserstein_kernel, \
    compute_dgm_from_graph
from tda.graph_dataset import DatasetLine
from tda.models import Architecture


class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"
    LastLayerSortedLogits = "LastLayerSortedLogits"


class KernelType(object):
    Euclidean = "Euclidean"
    RBF = "RBF"
    SlicedWasserstein = "SlicedWasserstein"


def get_embedding(
        embedding_type: str,
        line: DatasetLine,
        architecture: Architecture,
        thresholds: Dict,
        params: Dict = dict()
):
    graph = Graph.from_architecture_and_data_point(
        architecture=architecture,
        x=line.x.double(),
        thresholds=thresholds
    )

    if embedding_type == EmbeddingType.AnonymousWalk:
        walk = AnonymousWalks(G=graph.to_nx_graph())
        embedding = walk.embed(
            steps=params['steps'],
            method="sampling",
            verbose=False)[0]
        return embedding
    elif embedding_type == EmbeddingType.WeisfeilerLehman:
        return get_wl_embedding(
            graph=graph,
            height=params['height'],
            hash_size=params['hash_size'],
            node_labels=params["node_labels"]
        ).todense()
    elif embedding_type == EmbeddingType.PersistentDiagram:
        return compute_dgm_from_graph(graph)
    elif embedding_type == EmbeddingType.LastLayerSortedLogits:
        return sorted(graph.final_logits)


def get_gram_matrix(
        kernel_type: str,
        embeddings_in: List,
        embeddings_out: Optional[List] = None,
        params: Dict = dict()
):
    """
    Compute the gram matrix of the given embeddings
    using a specific kernel

    :param kernel_type: Kernel to be used (str)
    :param embeddings_in: Embeddings in
    :param embeddings_out: Embeddings out
        if None, we use the same as embeddings_in and use the fact
        the matrix is symmetric to speed up the process
    :param params: Parameters for the kernel
    :return:
    """

    n = len(embeddings_in)

    # If no embeddings out is specified, we use embeddings_in
    if not embeddings_out:
        embeddings_out = embeddings_in
        gram = np.zeros([n, n])
        sym = True
    else:
        m = len(embeddings_out)
        gram = np.zeros([n, m])
        sym = False

    for i in range(n):

        if sym:
            lim = i+1
        else:
            lim = m

        for j in range(lim):
            if kernel_type == KernelType.Euclidean:
                gram[i, j] = np.transpose(embeddings_in[i])@embeddings_out[j]
            elif kernel_type == KernelType.RBF:
                gram[i, j] = np.exp(-np.linalg.norm(
                    embeddings_in[i] - embeddings_out[j]
                ) / 2 * params['gamma']**2)
            elif kernel_type == KernelType.SlicedWasserstein:
                sw = sliced_wasserstein_kernel(
                    embeddings_in[i],
                    embeddings_out[j],
                    M=params['M'])
                gram[i, j] = np.exp(-sw / (2 * params['sigma'] ** 2))
            else:
                raise NotImplementedError(
                    f"Unknown kernel {kernel_type}"
                )
            if sym:
                gram[j, i] = gram[i, j]
    return gram