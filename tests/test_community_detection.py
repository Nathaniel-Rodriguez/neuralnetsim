import unittest
import neuralnetsim
from pkg_resources import resource_filename
from pkg_resources import resource_isdir
from pathlib import Path


class TestCommunityDetection(unittest.TestCase):
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
        self.assertEqual(len({m for i, m in graph.nodes.data("level1")}), 74)
        self.assertEqual(len({m for i, m in graph.nodes.data("level2")}), 57)

    def test_add_positions(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        graph = neuralnetsim.add_positions(graph, self.xpos, self.ypos)
        self.assertAlmostEqual(graph.nodes[0]["pos"][0], 446.5, 1)
        self.assertAlmostEqual(graph.nodes[0]["pos"][1], -45.01, 1)

    def test_build_graph_from_data(self):
        self.assertTrue(resource_isdir("tests", "test_data"))
        data_dir = resource_filename("tests", "test_data")
        graph = neuralnetsim.build_graph_from_data(
            Path(data_dir),
            "pdf",
            "weights",
            "xy"
        )
        self.assertSetEqual(set(graph.nodes[0].keys()), {'level1', 'level2', 'pos'})
        self.assertEqual(graph.nodes[0]['level1'], 20)
        self.assertEqual(graph.nodes[0]['level2'], 20)
        self.assertEqual(len(graph.nodes[0]['pos']), 2)
