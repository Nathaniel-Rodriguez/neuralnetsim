__all__ = []


import numpy as np
import networkx as nx
import random
from typing import Callable
from neuralnetsim import CoolingSchedule


def create_bridge_mask(graph: nx.DiGraph, key: str) -> np.ndarray:
    """
    Creates a numpy mask which is true for connections that are bridges.
    :param graph: A networkx graph.
    :param key: A key specifying the community node attribute.
    :return: A numpy array of size NxN, where N is the number of nodes in the
            graph.
    """
    # Construct matrix of 1s for bridge 0s for com
    bridge_matrix = np.zeros((len(graph), len(graph)))
    for node_i in graph.nodes:
        for node_j in graph.nodes:
            # check if edge is within or between communities
            if graph.nodes[node_i][key] == graph.nodes[node_j][key]:
                bridge_matrix[node_i, node_j] = 0
            else:
                bridge_matrix[node_i, node_j] = 1

    return bridge_matrix != 0


def create_log_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Creates a copy of the adjacency matrix with log applied element-wise.
    Zero entries are excluded.
    :param adj_matrix: An adjacency matrix.
    :return: The logged matrix copy.
    """
    log_matrix = adj_matrix.copy()
    non_zero_mask = adj_matrix != 0
    log_matrix[non_zero_mask] = np.log(adj_matrix[non_zero_mask])
    return log_matrix


class NetworkAnnealer:
    def __init__(self,
                 num_steps: int,
                 num_scramble_steps: int,
                 cooling_schedule: CoolingSchedule,
                 energy_function: Callable[[np.ndarray], float],
                 seed: int = None):
        if seed is None:
            self._rng = random.Random()
        else:
            self._rng = random.Random(seed)
        self._num_steps = num_steps
        self._num_scramble_steps = num_scramble_steps

    def _swap(self):
        pass

    def _step(self):
        pass

    def fit(self, graph: nx.DiGraph, community_key: str, **parameters):
        num_edges = graph.number_of_edges()
        num_nodes = graph.number_of_nodes()
        adj_mat = nx.to_numpy_matrix(graph).getA()
        bridge_mask = create_bridge_mask(graph, community_key)
        log_mat = create_log_matrix(adj_mat)

        return self

    def predict(self) -> nx.DiGraph:
        pass

    def fit_predict(self, graph: nx.DiGraph) -> nx.DiGraph:
        return self.fit(graph).predict()
