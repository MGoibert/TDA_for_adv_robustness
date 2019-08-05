import typing
from tda.graph import Graph
from tda.embeddings.anonymous_walk import AnonymousWalks
from tda.embeddings.weisfeiler_lehman import get_wl_embedding
from tda.embeddings.persistent_diagrams import compute_dgm_from_edges


class EmbeddingType(object):
    AnonymousWalk = "AnonymousWalk"
    WeisfeilerLehman = "WeisfeilerLehman"
    PersistentDiagram = "PersistentDiagram"


def get_embedding(
        embedding_type: str,
        graph: Graph,
        params: typing.Dict
):
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
