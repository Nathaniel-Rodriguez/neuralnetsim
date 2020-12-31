import unittest
import neuralnetsim
import networkx as nx
import numpy as np
from pkg_resources import resource_filename
from pkg_resources import resource_isdir
from pathlib import Path


class TestNetworkPreprocessing(unittest.TestCase):
    def setUp(self):
        self.link_path = resource_filename("tests", "test_data/pdf.mat")
        self.weight_path = resource_filename("tests", "test_data/weights.mat")
        self.pos_path = resource_filename("tests", "test_data/xy.mat")
        self.link_mat = neuralnetsim.load_as_matrix(self.link_path, "pdf")
        self.weight_mat = neuralnetsim.load_as_matrix(self.weight_path, "weights")
        self.xpos = neuralnetsim.load_as_matrix(self.pos_path, "x")
        self.ypos = neuralnetsim.load_as_matrix(self.pos_path, "y")

    def tearDown(self):
        pass

    def test_get_network(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        self.assertEqual(len(graph.nodes), 243)
        self.assertEqual(len(graph.edges), 1098)

    def test_add_communities(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        neuralnetsim.add_communities(graph, seed=4953)
        for node in graph.nodes:
            self.assertTrue("level1" in graph.nodes[node])
            self.assertTrue("level2" in graph.nodes[node])

    def test_add_positions(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        graph = neuralnetsim.add_positions(graph, self.xpos, self.ypos)
        self.assertAlmostEqual(graph.nodes[0]["pos"][0], 446.5, 1)
        self.assertAlmostEqual(graph.nodes[0]["pos"][1], -45.01, 1)
        for node in graph.nodes:
            self.assertEqual(len(graph.nodes[node]['pos']), 2)

    def test_apply_weight_threshold(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        graph = neuralnetsim.apply_weight_threshold(graph)
        self.assertEqual(nx.number_weakly_connected_components(graph), 1)

        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        graph = neuralnetsim.apply_weight_threshold(graph, 200.0)
        self.assertEqual(len(graph), 1)

        graph = nx.DiGraph()
        graph.add_edge(1, 2, weight=0.1)
        graph.add_edge(1, 3, weight=0.2)
        graph.add_edge(1, 4, weight=0.3)
        graph = neuralnetsim.apply_weight_threshold(graph, 0.0)
        self.assertEqual(len(graph), 4)
        graph = neuralnetsim.apply_weight_threshold(graph, 0.15)
        self.assertEqual(len(graph), 3)

    def test_build_graph_from_data(self):
        self.assertTrue(resource_isdir("tests", "test_data"))
        data_dir = resource_filename("tests", "test_data")
        graph = neuralnetsim.build_graph_from_data(
            Path(data_dir),
            "pdf",
            "weights",
            "xy",
        )
        self.assertSetEqual(set(k for node in graph.nodes
                                for k in graph.nodes[node].keys()),
                            {'level1', 'level2', 'pos'})
        for node in graph.nodes:
            self.assertEqual(len(graph.nodes[node]['pos']), 2)

    def test_get_first_loss_graph(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 1, weight=0.1)  # should be removed
        graph.add_edge(0, 2, weight=0.25)
        graph.add_edge(1, 2, weight=0.2)  # should be removed
        graph.add_edge(1, 3, weight=0.27)
        graph.add_edge(2, 3, weight=0.3)
        graph.add_edge(3, 4, weight=0.4)

        test_graph = neuralnetsim.get_first_loss_graph(graph)
        self.assertEqual(len(test_graph), 5)
        self.assertEqual(nx.number_of_edges(test_graph), 4)
        edges = [(0, 2), (1, 3), (2, 3), (3, 4)]
        for edge in test_graph.edges():
            self.assertTrue(edge in edges)

    def test_convert_scale(self):
        graph = nx.DiGraph()
        graph.add_edge(0, 2, weight=0.9)
        graph.add_edge(1, 3, weight=0.2)
        test_graph = neuralnetsim.convert_scale(graph)
        correct = {(0, 2): np.fabs(1/np.log(0.9)),
                   (1, 3): np.fabs(1/np.log(0.2))}
        for edge, weight in nx.get_edge_attributes(test_graph, "weight").items():
            self.assertAlmostEqual(weight, correct[edge])
