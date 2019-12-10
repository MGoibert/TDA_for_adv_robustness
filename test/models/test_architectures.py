from tda.models import Architecture


def test_walk_through_dag_list():
    my_edges = [
        (1, 2), (2, 3), (3, 4)
    ]
    my_order = Architecture.walk_through_dag(my_edges)
    assert my_order == [1, 2, 3, 4]


def test_walk_through_dag_simple():
    my_edges = [
        (1, 2), (2, 3), (3, 4),
        (6, 5),
        (2, 5), (5, 4)  # Shortcut for layer 2 to 4
    ]
    my_order = Architecture.walk_through_dag(my_edges)

    # Check that all the nodes are visited
    for edge in my_edges:
        assert edge[0] in my_order and edge[1] in my_order

    # Check that for all edges the source is visited before the target
    for edge in my_edges:
        assert my_order.index(edge[0]) < my_order.index(edge[1])

    print(Architecture.get_parent_dict(my_edges))



