__all__ = ["calc_mu", "calc_strength_distribution",
           "calc_nodal_strength_difference_distribution",
           "get_communities"]


import numpy as np
import networkx as nx
from typing import List


def calc_mu(graph: nx.DiGraph, key: str) -> float:
    """
    Calculates the ratio of inter-community link strength to the total strength
    of links. The "weight" key is assumed to exist for edges.

    :param graph: A networkx graph.
    :param key: The node attribute community key.
    :return: The community strength, mu.
    """
    return sum(
        weight for edge, weight in nx.get_edge_attributes(graph, "weight").items()
        if graph.nodes[edge[0]][key] != graph.nodes[edge[1]][key]
    ) / sum(weight for edge, weight in nx.get_edge_attributes(graph, "weight").items())


def get_communities(graph: nx.DiGraph, key: str) -> List[int]:
    """

    :param graph:
    :param key:
    :return:
    """
    return sorted(list(set(com for com in nx.get_node_attributes(graph, key).values())))


def calc_strength_distribution(graph: nx.DiGraph, direction: str) -> np.ndarray:
    """
    Generates a array of in- or out- strengths for each node. This is the sum
    of incoming or outgoing weights into (or from) a node.

    :param graph: A networkx directed graph with "weight" edge attributes.
    :param direction: The direction to sum for the node. Either "in" or "out".
    :return: A array of strength values for each node in graph.nodes() order.
        This is the default ordering for networkx graphs.
    """
    if direction == "in":
        distribution = np.sum(nx.to_numpy_array(graph), axis=0)
        return distribution[distribution != 0.0].copy()
    elif direction == "out":
        distribution = np.sum(nx.to_numpy_array(graph), axis=1)
        return distribution[distribution != 0.0].copy()


def calc_nodal_strength_difference_distribution(graph: nx.DiGraph) -> np.ndarray:
    """
    Generates a array of in-strength vs out-strength differences for each
    node. This is the difference between the sum of incoming and outgoing
    weights.

    :param graph: A networkx directed graph with "weight" edge attributes.
    :return: An array of strength values for each node in graph.nodes() order.
        This is the default ordering for networkx graphs.
    """
    return np.sum(nx.to_numpy_array(graph)
                  - nx.to_numpy_array(graph).T, axis=1)
