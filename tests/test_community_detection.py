import unittest
import neuralnetsim
from pkg_resources import resource_filename


class TestCommunityDetection(unittest.TestCase):
    def setUp(self):
        link_path = resource_filename("tests", "test_data/pdf.mat")
        weight_path = resource_filename("tests", "test_data/weights.mat")
        self.link_mat = neuralnetsim.load_as_matrix(link_path, "pdf")
        self.weight_mat = neuralnetsim.load_as_matrix(weight_path, "weights")

    def tearDown(self):
        pass

    def test_get_network(self):
        graph = neuralnetsim.get_network(self.weight_mat, self.link_mat)
        self.assertEqual(len(graph.nodes), 243)
        self.assertEqual(len(graph.edges), 1098)
