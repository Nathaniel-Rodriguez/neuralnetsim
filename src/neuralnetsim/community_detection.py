__all__ = ["get_network", "add_communities"]


import infomap
import numpy as np
import networkx as nx


def get_network(weight_matrix: np.ndarray,
                link_matrix: np.ndarray) -> nx.DiGraph:
    """
    Creates a networkx graph from loaded data.
    :param weight_matrix: A matrix of weights.
    :param link_matrix: A matrix of significant links.
    :return: A directed networkx graph.
    """
    adj_matrix = weight_matrix.copy()
    mask = np.array(link_matrix, dtype=np.bool)
    adj_matrix[np.where(~mask)] = 0
    return nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)


def add_communities(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Uses Infomap to detect communities in a given graph and then assigns
    those communities as an attribute to the nodes of graph called "community".
    :param graph: A weighted-directed networkx graph.
    :return: Updates graph in-place with community assignments, returns the
             provided graph.
    """
    imap = infomap.Infomap("-N 10 --two-level --directed")
    imap.add_links(tuple((i, j, w) for i, j, w in graph.edges.data('weight')))
    imap.run()
    nx.set_node_attributes(graph,
                           {node.node_id: node.module_id
                            for node in imap.tree if node.is_leaf},
                           "community")
    return graph
