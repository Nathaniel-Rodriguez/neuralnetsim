import unittest
import networkx as nx
import numpy as np
import neuralnetsim


class HelperEnergyFunction:
    def __init__(self, target_graph):
        self._target_adj = nx.to_numpy_array(target_graph)

    def __call__(self, matrix):
        return np.sum(np.abs(self._target_adj - matrix))


def scramble_graph(matrix, rng):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x = rng.randint(0, matrix.shape[0])
            y = rng.randint(0, matrix.shape[0])
            if x != y:
                matrix[i, j], matrix[x, y] = matrix[x, y], matrix[i, j]


class TestNetworkAnnealerDebug(unittest.TestCase):
    def test_annealing(self):
        rng = np.random.RandomState(593)
        target_mat = rng.normal(size=(5, 5))
        target_mat[target_mat < 0] = 0
        np.fill_diagonal(target_mat, 0)
        target_graph = nx.from_numpy_array(target_mat,
                                           create_using=nx.DiGraph)
        original_mat = target_mat.copy()
        scramble_graph(original_mat, rng)
        original_graph = nx.from_numpy_array(original_mat,
                                             create_using=nx.DiGraph)

        cooling = neuralnetsim.AdaptiveCoolingSchedule(t0=1.0,
                                                       cooling_factor=0.1,
                                                       max_estimate_window=20,
                                                       hold_window=10)
        efunc = HelperEnergyFunction(target_graph)
        annealer = neuralnetsim.NetworkAnnealerDebug(
            2000, 10, cooling, efunc, 53)
        result_graph = annealer.fit_predict(original_graph)
        result_mat = nx.to_numpy_array(result_graph)
        self.assertTrue(np.all(result_mat == target_mat))
