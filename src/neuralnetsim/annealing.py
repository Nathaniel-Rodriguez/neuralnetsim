__all__ = ["NetworkAnnealer"]


import numpy as np
import networkx as nx
import random
import math
from typing import Callable
from typing import Tuple
from typing import List
from neuralnetsim import CoolingSchedule
from collections import namedtuple


SwapResult = namedtuple("SwapResult", ["energy", "source_index", "source_edge",
                                       "destination_edge", "destination_exists"])


def edge_swap(matrix: np.ndarray,
              edges: List[Tuple[int, int]],
              source_index: int,
              source_edge: Tuple[int, int],
              destination_edge: Tuple[int, int],
              destination_exists: bool) -> np.ndarray:
    # swap elements in matrix
    matrix[source_edge[0], source_edge[1]], \
    matrix[destination_edge[0], destination_edge[1]] = \
        matrix[destination_edge[0], destination_edge[1]], \
        matrix[source_edge[0], source_edge[1]]

    # Create new edge if destination doesn't exist
    if not destination_exists:
        edges[source_index] = (destination_edge[0], destination_edge[1])

    return matrix


class NetworkAnnealer:
    """
    Generates a new network from a given one by minimizing a given energy
    function.
    """
    def __init__(self,
                 num_steps: int,
                 num_scramble_steps: int,
                 cooling_schedule: CoolingSchedule,
                 energy_function: Callable[[np.ndarray], float],
                 seed: int = None):
        """
        :param num_steps: The number of steps for the annealing process.
        :param num_scramble_steps: The number of steps that will be spent
        scrambling the network before annealing begins.
        :param cooling_schedule: A cooling schedule for the annealing process.
        The cooling schedule only activates following the scrambling period.
        :param energy_function: An energy function that can be evaluated
        when provided an adjacency matrix of the current network.
        :param seed: A seed for random number generation.
        """
        if seed is None:
            self._rng = random.Random()
        else:
            self._rng = random.Random(seed)
        self.num_steps = num_steps
        self.num_scramble_steps = num_scramble_steps
        self._cooling_schedule = cooling_schedule
        self._energy_function = energy_function
        self._energy = None
        self._history = []
        self._acceptance_history = []
        self._graph = None

    def _scramble(self, num_edges, num_nodes, adj_mat, edges):
        # Pick random edge to move
        source_index = self._rng.randrange(num_edges)
        source_edge = edges[source_index]
        # Pick random destination (prevent self-loops)
        destination_edge = (self._rng.randrange(num_nodes),
                            self._rng.randrange(num_nodes))
        # keep redrawing if a self-loop is drawn
        while destination_edge[0] == destination_edge[1]:
            destination_edge = (self._rng.randrange(num_nodes),
                                self._rng.randrange(num_nodes))
        destination_exists = destination_edge in edges
        energy = self._energy_function(
            edge_swap(adj_mat, edges, source_index, source_edge,
                      destination_edge, destination_exists))

        return SwapResult(energy, source_index, source_edge,
                          destination_edge, destination_exists)

    def _anneal(self, adj_mat, edges, energy, source_index, source_edge,
                destination_edge, destination_exists):
        energy_diff = energy - self._energy
        if energy_diff < 0:
            self._acceptance_history.append(1)
            accepted_energy = energy
        else:
            if self._rng.random() < math.exp(-energy_diff / self._cooling_schedule.t):
                accepted_energy = energy
                self._acceptance_history.append(1)
            else:
                self._acceptance_history.append(0)
                edge_swap(adj_mat, edges, source_index,
                          destination_edge, source_edge, destination_exists)
                accepted_energy = self._energy
        self._cooling_schedule.step(accepted_energy)
        return accepted_energy

    def fit(self, graph: nx.DiGraph):
        self._graph = graph
        num_edges = self._graph.number_of_edges()
        num_nodes = self._graph.number_of_nodes()
        adj_mat = nx.to_numpy_array(self._graph)
        edges = np.argwhere(adj_mat).tolist()
        self._energy = self._energy_function(adj_mat)
        self._history.append(self._energy)

        for i in range(self.num_scramble_steps):
            self._energy = self._scramble(num_edges, num_nodes,
                                          adj_mat, edges).energy
            self._history.append(self._energy)

        for i in range(self.num_steps):
            self._energy = self._anneal(
                adj_mat, edges, *self._scramble(
                    num_edges, num_nodes, adj_mat, edges))
            self._history.append(self._energy)

        return self

    def predict(self) -> nx.DiGraph:
        pass

    def fit_predict(self, graph: nx.DiGraph) -> nx.DiGraph:
        return self.fit(graph).predict()

    @property
    def cooling_scheduler(self):
        return self._cooling_schedule

    @cooling_scheduler.setter
    def cooling_scheduler(self, value):
        raise NotImplementedError

    @property
    def energy_function(self):
        return self._energy_function

    @energy_function.setter
    def energy_function(self, value):
        raise NotImplementedError

    @property
    def energy_history(self):
        return self._history

    @energy_history.setter
    def energy_history(self, v):
        raise NotImplementedError

    @property
    def acceptance_history(self):
        return self._acceptance_history

    @acceptance_history.setter
    def acceptance_history(self, v):
        raise NotImplementedError
