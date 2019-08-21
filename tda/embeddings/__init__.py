import typing
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import compute_dgm_from_edges

try:
    import dionysus
except Exception as e:
    print("Unable to find dionysus")
    dionysus = None

class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"
    OriginalDataPoint = "OriginalDataPoint"


Embedding = object
Kernel = typing.Optional[typing.Callable]


def get_embedding(
        embedding_type: str,
        graph: Graph,
        params: typing.Dict
) -> Embedding:
    if embedding_type == EmbeddingType.AnonymousWalk:
        walk = AnonymousWalks(G=graph.to_nx_graph(params['threshold']))
        embedding = walk.embed(
            steps=params['steps'],
            method="sampling",
            verbose=False)[0]
        return embedding
    elif embedding_type == EmbeddingType.WeisfeilerLehman:
        return get_wl_embedding(
            graph=graph.to_nx_graph(params['threshold']),
            height=params['height'],
            hash_size=params['hash_size']).todense()
    elif embedding_type == EmbeddingType.PersistentDiagram:
        return compute_dgm_from_edges(graph._edge_dict, params['threshold'])
    elif embedding_type == EmbeddingType.OriginalDataPoint:
        return graph.original_data_point


def get_kernel(embedding_type: str) -> Kernel:
    if embedding_type == EmbeddingType.PersistentDiagram:
        def my_kernel(x, y):
            return dionysus.wasserstein_distance(x, y, q=2)
        return my_kernel
    return None
