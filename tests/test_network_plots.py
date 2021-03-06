import unittest
import neuralnetsim
import numpy as np
from pkg_resources import resource_filename
from pathlib import Path


class TestNetworkPlots(unittest.TestCase):
    def setUp(self):
        self.link_path = resource_filename("tests", "test_data/pdf.mat")
        self.weight_path = resource_filename("tests", "test_data/weights.mat")
        self.pos_path = resource_filename("tests", "test_data/xy.mat")
        self.link_mat = neuralnetsim.load_as_matrix(self.link_path, "pdf")
        self.weight_mat = neuralnetsim.load_as_matrix(self.weight_path, "weights")
        self.xpos = neuralnetsim.load_as_matrix(self.pos_path, "x")
        self.ypos = neuralnetsim.load_as_matrix(self.pos_path, "y")

    def test_plot_slice(self):
        data_dir = resource_filename("tests", "test_data")
        graph = neuralnetsim.build_graph_from_data(
            Path(data_dir),
            "pdf",
            "weights",
            "xy"
        )
        neuralnetsim.plot_slice(graph, color_key="level2", prefix="test")
        # plot has to be visually inspected
        self.assertTrue(True)

    def test_plot_ccdf_distributions(self):
        rng = np.random.RandomState(80234)
        original = rng.lognormal(size=200)
        others = [rng.lognormal(size=200) for i in range(10)]
        neuralnetsim.plot_ccdf_distributions(original, others, "testx", "testy",
                                             "log", "linear", prefix="test")
        # plot has to be visually inspected
        self.assertTrue(True)
