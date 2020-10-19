__all__ = ["ModularityEnergyFunction",
           "StrengthDistributionEnergyFunction",
           "NeuronalStrengthDifferenceEnergyFunction",
           "NeuralEnergyFunction"]


import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance
from neuralnetsim import create_bridge_mask
from neuralnetsim import create_log_matrix


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

    def __call__(self, matrix: np.ndarray) -> float:
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
    """
    Evaluates the network energy associated with the nodal strength distribution
    for in-degree and out-degree. The energy is calculated using the
    Wasserstein distance applied to the logarithm of the strengths.
    """
    def __init__(self, graph: nx.DiGraph):
        """
        :param graph: The original graph that will be used as the basis for
        assessing the target in-strength distribution and out-strength
        distribution.
        """
        self._target_in_strength_distribution = \
            np.sum(nx.to_numpy_array(graph), axis=0)
        self._target_out_strength_distribution = \
            np.sum(nx.to_numpy_array(graph), axis=1)

    def __call__(self, matrix: np.ndarray) -> float:
        """
        Evaluates the energy.
        :param matrix: An logged or non-logged adjacency matrix.
        :return: The energy of the matrix.
        """
        in_strength_distribution = np.sum(matrix, axis=0)
        out_strength_distribution = np.sum(matrix, axis=1)
        return wasserstein_distance(in_strength_distribution,
                                    self._target_in_strength_distribution) \
               + wasserstein_distance(out_strength_distribution,
                                      self._target_out_strength_distribution)


class NeuronalStrengthDifferenceEnergyFunction:
    """
    Evaluate the network energy associated with the neural differences in
    in- and out- strength of a given neuron. These two values are close to the
    same for any given neuron. In order to preserve this nodal associativity
    the difference between neuron in-strengths and out-strengths is taken.
    If near zero, then it matches with expectations.
    """
    def __call__(self, matrix: np.ndarray) -> float:
        """
        Evaluates the energy.
        :param matrix: An logged or non-logged adjacency matrix.
        :return: The energy of the matrix.
        """
        return np.mean(np.power(np.sum(matrix - matrix.T, axis=1), 2))


class NeuralEnergyFunction:
    def __init__(self,
                 modularity_energy_function: ModularityEnergyFunction,
                 strength_dist_energy_function: StrengthDistributionEnergyFunction,
                 strength_difference_energy_function: NeuronalStrengthDifferenceEnergyFunction,
                 ):
        pass

    def __call__(self, matrix: np.ndarray) -> float:
        pass
