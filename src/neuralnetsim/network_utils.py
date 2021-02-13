__all__ = ["create_bridge_mask",
           "create_log_matrix"]


import numpy as np
import networkx as nx


def create_bridge_mask(graph: nx.DiGraph, key: str) -> np.ndarray:
    """
    Creates a numpy mask which is true for connections that are bridges.

    :param graph: A networkx graph.
    :param key: A key specifying the community node attribute.
    :return: A numpy bool array of size NxN, where N is the number of nodes in
            the graph.
    """
    # Construct matrix of 1s for bridge 0s for com
    bridge_matrix = np.zeros((len(graph), len(graph)), dtype=np.bool)
    for i, node_i in enumerate(graph.nodes):
        for j, node_j in enumerate(graph.nodes):
            # check if edge is within or between communities
            if graph.nodes[node_i][key] == graph.nodes[node_j][key]:
                bridge_matrix[i, j] = 0
            else:
                bridge_matrix[i, j] = 1

    return bridge_matrix != 0


def create_log_matrix(adj_matrix: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    """
    Creates a an adjacency matrix with log applied element-wise.
    Zero entries are excluded. Does not raise exceptions for negative entries.
    Check for NaN in these cases.

    :param adj_matrix: An adjacency matrix.
    :param out: An optional output matrix (default: None).
    :return: The logged matrix. If out is not specified a copy is created.
    """
    if out is None:
        log_matrix = adj_matrix.copy()
    else:
        log_matrix = out
        log_matrix[:] = adj_matrix[:]
    non_zero_mask = adj_matrix != 0
    log_matrix[non_zero_mask] = np.log(adj_matrix[non_zero_mask])
    return log_matrix
