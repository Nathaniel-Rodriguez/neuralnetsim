import unittest
import neuralnetsim
import networkx as nx


class TestModularityEnergyFunction(unittest.TestCase):
    def test_energy_evaluation(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=2)
        graph.add_node(4, com=2)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(3, 4, weight=1)
        graph.add_edge(2, 3, weight=1)

        efunc = neuralnetsim.ModularityEnergyFunction(graph, 0.0, "com")
        mat = nx.to_numpy_array(graph)
        self.assertAlmostEqual(efunc(mat), (0. - 1. / 3.)**2)


class TestStrengthDistributionEnergyFunction(unittest.TestCase):
    def test_energy_evaluation(self):
        graph = nx.DiGraph()
        graph.add_node(1, com=1)
        graph.add_node(2, com=1)
        graph.add_node(3, com=2)
        graph.add_node(4, com=2)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(3, 4, weight=1)
        graph.add_edge(2, 3, weight=1)

        efunc = neuralnetsim.StrengthDistributionEnergyFunction(graph)
        mat = nx.to_numpy_array(graph)
        self.assertAlmostEqual(efunc(mat), 0.0)
        mat[0, 2] = 1
        self.assertGreater(efunc(mat), 0.0)


class TestNeuronalStrengthDifferenceEnergyFunction(unittest.TestCase):
    def test_energy_evaluation(self):
        graph = nx.DiGraph()
        graph.add_node(1)
        graph.add_node(2)
        graph.add_node(3)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 1, weight=1)
        graph.add_edge(2, 3, weight=1)
        graph.add_edge(3, 2, weight=1)
        graph.add_edge(3, 1, weight=1)
        graph.add_edge(1, 3, weight=1)

        efunc = neuralnetsim.NeuronalStrengthDifferenceEnergyFunction()
        mat = nx.to_numpy_array(graph)
        self.assertAlmostEqual(efunc(mat), 0.0)
        graph.remove_edge(1, 2)
        mat = nx.to_numpy_array(graph)
        self.assertGreater(efunc(mat), 0.0)
