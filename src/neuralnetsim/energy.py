__all__ = ["ModularityEnergyFunction",
           "StrengthDistributionEnergyFunction",
           "NeuronalStrengthDifferenceEnergyFunction",
           "NeuralEnergyFunction",
           "HistoryMixin",
           "ModularityEnergyFunctionDebug",
           "StrengthDistributionEnergyFunctionDebug",
           "NeuronalStrengthDifferenceEnergyFunctionDebug",
           "NeuralEnergyFunctionDebug"]


import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance
from neuralnetsim import create_bridge_mask


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

    def __call__(self, matrix: np.ndarray) -> float:
        """
        Calculates the new energy given an adjacency matrix. Uses the square
        of the difference between the current and target modularity.
        :param matrix: The new graphs adjacency matrix. Nodes must correspond
        to those of the original provided graph, edges can vary.
        :return: The energy of the network.
        """
        modularity = np.sum(matrix[self._bridge_mask]) / self._total_edge_weight
        return (self.target_modularity - modularity) ** 2


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

    # noinspection PyTypeChecker
    def __call__(self, matrix: np.ndarray) -> float:
        """
        Evaluates the energy.
        :param matrix: An logged or non-logged adjacency matrix.
        :return: The energy of the matrix.
        """
        return np.mean(np.power(np.sum(matrix - matrix.T, axis=1), 2))


class NeuralEnergyFunction:
    """
    Evaluates all three network energy functions: Modularity,
     strength distributions, and in- out- strength.
    """
    def __init__(self,
                 graph: nx.DiGraph,
                 target_modularity: float,
                 community_key: str,
                 modularity_weight: float = 1.0,
                 strength_dist_weight: float = 1.0,
                 strength_difference_weight: float = 1.0):
        """
        :param graph: The original graph that the energy functions will be
        applied too.
        :param target_modularity: The target modularity.
        :param community_key: The graph node key for communities.
        :param modularity_weight: Scale factor for modularity energy.
        :param strength_dist_weight: Scale factor for strength distribution
        energy.
        :param strength_difference_weight: Scale factor for in/out node strength
        difference energy.
        """
        self.modularity_weight = modularity_weight
        self.strength_dist_weight = strength_dist_weight
        self.strength_difference_weight = strength_difference_weight
        self.modularity_energy_function = ModularityEnergyFunction(
            graph, target_modularity, community_key)
        self.strength_dist_energy_function = StrengthDistributionEnergyFunction(graph)
        self.strength_difference_energy_function = NeuronalStrengthDifferenceEnergyFunction()

    def __call__(self, matrix: np.ndarray) -> float:
        """
        Evaluates the energy.
        :param matrix: An adjacency matrix.
        :return: The energy of the network.
        """
        return self.modularity_weight * self.modularity_energy_function(matrix) \
               + self.strength_dist_weight * self.strength_dist_energy_function(matrix) \
               + self.strength_difference_weight * self.strength_difference_energy_function(matrix)


class HistoryMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = []

    # noinspection PyUnresolvedReferences
    def __call__(self, matrix: np.ndarray) -> float:
        energy = super().__call__(matrix)
        self._history.append(energy)
        return energy

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, v):
        raise NotImplementedError


class ModularityEnergyFunctionDebug(HistoryMixin, ModularityEnergyFunction):
    pass


class StrengthDistributionEnergyFunctionDebug(HistoryMixin, StrengthDistributionEnergyFunction):
    pass


class NeuronalStrengthDifferenceEnergyFunctionDebug(HistoryMixin, NeuronalStrengthDifferenceEnergyFunction):
    pass


class NeuralEnergyFunctionDebug(HistoryMixin, NeuralEnergyFunction):
    def __int__(self, graph: nx.DiGraph,
                target_modularity: float,
                community_key: str,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modularity_energy_function = ModularityEnergyFunctionDebug(
            graph, target_modularity, community_key)
        self.strength_dist_energy_function = StrengthDistributionEnergyFunctionDebug(graph)
        self.strength_difference_energy_function = NeuronalStrengthDifferenceEnergyFunctionDebug()
