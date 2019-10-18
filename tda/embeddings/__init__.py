import typing
import numpy as np
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import compute_dgm_from_edges, \
    sliced_wasserstein_kernel


class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"
    OriginalDataPoint = "OriginalDataPoint"
    LastLayerSortedLogits = "LastLayerSortedLogits"


class KernelType(object):
    Euclidean = "Euclidean"
    RBF = "RBF"
    SlicedWasserstein = "SlicedWasserstein"


def get_embedding(
        embedding_type: str,
        graph: Graph,
        params: typing.Dict
):
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
        return compute_dgm_from_edges(graph._edge_list)
    elif embedding_type == EmbeddingType.OriginalDataPoint:
        return np.reshape(graph.original_data_point, (-1))
    elif embedding_type == EmbeddingType.LastLayerSortedLogits:
        return sorted(graph.final_logits)


def get_gram_matrix(
        kernel_type: str,
        embeddings_in: typing.List,
        embeddings_out: typing.List,
        params: typing.Dict = dict()
):
    n = len(embeddings_in)
    m = len(embeddings_out)
    gram = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if kernel_type == KernelType.Euclidean:
                gram[i, j] = np.dot(
                    embeddings_in[i],
                    embeddings_out[j]
                )
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
                    f"Unknown kenerl {kernel_type}"
                )
    return gram