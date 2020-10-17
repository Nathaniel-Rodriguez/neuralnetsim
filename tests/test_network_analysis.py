import unittest
import neuralnetsim
import networkx as nx


class TestNetworkAnalysis(unittest.TestCase):
    def test_calc_mu(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(4, com=2)
        graph.add_node(5, com=3)
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(2, 4, weight=1.0)
        graph.add_edge(4, 5, weight=1.5)
        graph.add_edge(5, 1, weight=2.0)
        self.assertAlmostEqual(neuralnetsim.calc_mu(graph, "com"),
                               (1.0 + 1.5 + 2.0) / (1.0 + 1.0 + 1.0 + 1.5 + 2.0))
