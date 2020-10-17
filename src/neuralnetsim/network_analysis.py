__all__ = ["calc_mu"]


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
