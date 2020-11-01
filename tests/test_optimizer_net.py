import unittest
import neuralnetsim
import networkx as nx
import nest
import numpy as np


class TestSubNetwork(unittest.TestCase):
    def setUp(self):
        nest.SetKernelStatus({'grng_seed': 984,
                              'rng_seeds': [684]})
        nest.set_verbosity(0)
        graph = nx.DiGraph()
        graph.add_edge(2, 1, weight=10000.0)
        graph.add_edge(3, 1, weight=10000.0)
        graph.add_edge(4, 1, weight=10000.0)

        neuron_pars = {}
        syn_pars = {"model": "static_synapse", "weight": 1.0, "delay": 2.0}
        noise_pars = {"mean": 2.0, "std": 1.0, "dt": 0.1, "frequency": 10.0}
        self.net = neuralnetsim.SubNetwork(graph, 1, neuron_pars, syn_pars,
                                           noise_pars, "iaf_tum_2000", 0.8)

    def tearDown(self) -> None:
        nest.ResetKernel()

    def test_spike_output(self):
        nest.Simulate(50.0)
        self.assertEqual(len(self.net.get_spike_output()), 0)

    def test_input_signal(self):
        self.net.set_inputs([np.array([10.]), np.array([]), np.array([])])
        nest.Simulate(50.0)
        self.assertEqual(len(self.net.get_spike_output()), 1)
        self.assertAlmostEqual(self.net.get_spike_output()[0], 12.7, 2)
