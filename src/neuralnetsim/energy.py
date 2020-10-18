__all__ = ["ModularityEnergyFunction",
           "StrengthDistributionEnergyFunction",
           "NeuronalStrengthDifferenceEnergyFunction",
           "NeuralEnergyFunction",
           "CompositeEnergyFunction"]


import numpy as np
import networkx as nx
from neuralnetsim import create_bridge_mask
from neuralnetsim import create_log_matrix
from typing import List
from typing import Callable


class ModularityEnergyFunction:
    """
    Evaluates the energy associated with the modularity of the network.
    The closer the modularity of a given network to the given target modularity
    the lower the energy. Energy is evaluated as the square difference between
    target and network energy based on the node communities from the given
    graph.
    """
    def __init__(self, graph: nx.DiGraph, target_modularity: float,
                 community_key: str):
        """
        :param graph: A networkx graph with a communities assigned to nodes.
        Must have a "weight" attribute for edges.
        :param target_modularity: The desired modularity.
        :param community_key: The key for the community node attribute to use.
        """
        self.target_modularity = target_modularity
        self.community_key = community_key
        self._bridge_mask = create_bridge_mask(graph, community_key)
        self._total_edge_weight = sum(nx.get_edge_attributes(graph, "weight").values())
        self._history = []

    def __call__(self, matrix: np.ndarray, log_matrix: np.ndarray = None) -> float:
        """
        Calculates the new energy given an adjacency matrix. Uses the square
        of the difference between the current and target modularity.
        :param matrix: The new graphs adjacency matrix. Nodes must correspond
        to those of the original provided graph, edges can vary.
        :return: The energy of the network.
        """
        modularity = np.sum(matrix[self._bridge_mask]) / self._total_edge_weight
        self._history.append(modularity)
        return (self.target_modularity - modularity) ** 2

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, v):
        raise NotImplementedError


class StrengthDistributionEnergyFunction:
    def __init__(self, graph: nx.DiGraph):
        pass

    def __call__(self, matrix: np.ndarray, log_matrix: np.ndarray) -> float:
        pass


class NeuronalStrengthDifferenceEnergyFunction:
    def __init__(self, graph: nx.DiGraph):
        pass

    def __call__(self, matrix: np.ndarray, log_matrix: np.ndarray) -> float:
        pass


class NeuralEnergyFunction:
    def __init__(self,
                 modularity_energy_function: ModularityEnergyFunction,
                 strength_dist_energy_function: StrengthDistributionEnergyFunction,
                 strength_difference_energy_function: NeuronalStrengthDifferenceEnergyFunction,
                 ):
        pass

    def __call__(self, matrix: np.ndarray, log_matrix: np.ndarray) -> float:
        pass


class CompositeEnergyFunction:
    def __init__(self,
                 energy_functions: List[Callable[[np.ndarray, np.ndarray], float]],
                 weightings: List[float]):
        pass

    def __call__(self, matrix: np.ndarray, log_matrix: np.ndarray) -> float:
        pass
