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


def add_communities(graph: nx.DiGraph, seed=None,
                    infomap_commands=None) -> nx.DiGraph:
    """
    Uses Infomap to detect communities in a given graph and then assigns
    those communities as an attribute to the nodes of graph called "level1"
    for first depth level modules and "level2" for second depth level modules.
    :param graph: A weighted-directed networkx graph.
    :param seed: A seed for Infomap (default: None).
    :param infomap_commands: Optional command arguments for Infomap (default:
        ["-N 10", "--two-level", "--directed", "--silent"]).
    :return: Updates graph in-place with community assignments, returns the
             provided graph.
    """
    if infomap_commands is None:
        infomap_commands = ["-N 10", "--two-level", "--directed", "--silent"]
    if seed is not None:
        infomap_commands.append("--seed " + str(seed))
    imap = infomap.Infomap(" ".join(infomap_commands))
    imap.add_links(tuple((i, j, w) for i, j, w in graph.edges.data('weight')))
    imap.run()
    nx.set_node_attributes(graph, imap.get_modules(depth_level=1), "level1")
    nx.set_node_attributes(graph, imap.get_modules(depth_level=2), "level2")
    return graph
