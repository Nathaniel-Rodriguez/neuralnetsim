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

    def test_calc_strength_distribution_in(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=2)
        graph.add_node(4, com=3)
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(2, 3, weight=1.0)
        graph.add_edge(3, 4, weight=1.5)
        graph.add_edge(4, 1, weight=2.0)

        in_data = neuralnetsim.calc_strength_distribution(graph, "in")
        self.assertAlmostEqual(in_data[0], 3.0)
        self.assertAlmostEqual(in_data[1], 1.0)
        self.assertAlmostEqual(in_data[2], 1.0)
        self.assertAlmostEqual(in_data[3], 1.5)

    def test_calc_strength_distribution_out(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=2)
        graph.add_node(4, com=3)
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(2, 3, weight=1.0)
        graph.add_edge(3, 4, weight=1.5)
        graph.add_edge(4, 1, weight=2.0)

        out_data = neuralnetsim.calc_strength_distribution(graph, "out")
        self.assertAlmostEqual(out_data[0], 1.0)
        self.assertAlmostEqual(out_data[1], 2.0)
        self.assertAlmostEqual(out_data[2], 1.5)
        self.assertAlmostEqual(out_data[3], 2.0)

    def test_calc_nodal_strength_difference_distribution(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=2)
        graph.add_node(4, com=3)
        graph.add_edge(1, 2, weight=1.0)
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(2, 3, weight=1.0)
        graph.add_edge(3, 4, weight=1.5)
        graph.add_edge(4, 1, weight=2.0)

        out_data = neuralnetsim.calc_nodal_strength_difference_distribution(graph)
        # out - in
        self.assertAlmostEqual(out_data[0], -2.0)
        self.assertAlmostEqual(out_data[1], 1.0)
        self.assertAlmostEqual(out_data[2], 0.5)
        self.assertAlmostEqual(out_data[3], 0.5)
