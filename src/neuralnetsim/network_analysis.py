__all__ = ["calc_mu", "calc_strength_distribution"]


import numpy as np
import networkx as nx


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


def calc_strength_distribution(graph: nx.DiGraph, direction: str) -> np.ndarray:
    """
    Generates a list of in- or out- strengths for each node. This is the sum
    of incoming or outgoing weights into (or from) a node.
    :param graph: A networkx directed graph with "weight" edge attributes.
    :param direction: The direction to sum for the node. Either "in" or "out".
    :return: A list of strength values for each node in graph.nodes() order.
    This is the default ordering for networkx graphs.
    """
    if direction == "in":
        return np.sum(nx.to_numpy_array(graph), axis=0)
    elif direction == "out":
        return np.sum(nx.to_numpy_array(graph), axis=1)
