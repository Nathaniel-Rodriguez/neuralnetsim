import unittest
import neuralnetsim
import networkx as nx
import numpy as np


class TestNetworkUtils(unittest.TestCase):
    def test_create_bridge_mask(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=1)
        graph.add_node(4, com=2)
        graph.add_node(5, com=2)
        graph.add_node(6, com=2)
        mask = neuralnetsim.create_bridge_mask(graph, "com")
        # check that all bridges are true and intra-comms false
        # mask uses adjacency matrix so it goes by id order
        for i in range(6):
            for j in range(6):
                if (i in {0, 1, 2} and j in {0, 1, 2})\
                            or (i in {3, 4, 5} and j in {3, 4, 5}):
                    self.assertFalse(mask[i, j])
                else:
                    self.assertTrue(mask[i, j])

    def test_create_log_matrix(self):
        mat = np.array([[0.0, 1.0],
                        [10.0, 100.0]])
        log_mat = neuralnetsim.create_log_matrix(mat)
        self.assertAlmostEqual(log_mat[0, 0], 0.0)
        self.assertAlmostEqual(log_mat[0, 1], np.log(1.0))
        self.assertAlmostEqual(log_mat[1, 0], np.log(10.0))
        self.assertAlmostEqual(log_mat[1, 1], np.log(100.0))
