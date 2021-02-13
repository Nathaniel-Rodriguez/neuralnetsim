import unittest
import neuralnetsim
import networkx as nx
import nest
import numpy as np


class TestNeuralCircuit(unittest.TestCase):
    def setUp(self) -> None:
        nest.set_verbosity(40)
        graph = nx.DiGraph()
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(1, 3, weight=1.0)
        graph.add_edge(3, 1, weight=1.0)
        self.params = neuralnetsim.CircuitParameters(
            graph,
            "iaf_tum_2000",
            {"t_ref_tot": 10.0, "t_ref_abs": 10.0},
            {"delay": 10.0},
            {"rate": 1000.0},
            {"weight_scale": 8000.0}
        )

    def test_setup(self):
        net = neuralnetsim.NeuralCircuit(self.params)
        self.assertEqual(len(net._neurons), 3)
        self.assertEqual(len(net._detectors), 3)
        self.assertEqual(nest.GetStatus(net._neurons)[0]['model'],
                         "iaf_tum_2000")
        self.assertAlmostEqual(
            nest.GetStatus(net._neurons)[0]['t_ref_tot'],
            10.0
        )
        self.assertAlmostEqual(
            nest.GetStatus(
                nest.GetConnections([net._neurons[0]],
                                    [net._neurons[1]]))[0]['weight'],
            8000.0
        )

    def test_run(self):
        net = neuralnetsim.NeuralCircuit(self.params)
        net.run(500.0)
        data = net.get_spike_trains()
        for spikes in data.values():
            self.assertGreater(len(spikes), 1)


class TestCircuitManager(unittest.TestCase):
    def setUp(self) -> None:
        nest.set_verbosity(40)
        graph = nx.DiGraph()
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(1, 3, weight=1.0)
        graph.add_edge(3, 1, weight=1.0)
        self.params = neuralnetsim.CircuitParameters(
            graph,
            "iaf_tum_2000",
            {"t_ref_tot": 10.0, "t_ref_abs": 10.0},
            {"delay": 10.0},
            {"rate": 1000.0},
            {"weight_scale": 8000.0}
        )

    def test_context(self):
        run_success = False
        with neuralnetsim.CircuitManager(
                neuralnetsim.NeuralCircuit,
                {'grng_seed': 16, 'rng_seeds': [24]},
                self.params) as net:
            net.run(500.0)
            data = net.get_spike_trains()
            for spikes in data.values():
                self.assertGreater(len(spikes), 1)
            run_success = True
        self.assertTrue(run_success)
