import unittest
import neuralnetsim
import networkx as nx
import nest
import numpy as np


class TestSubNetwork(unittest.TestCase):
    def setUp(self):
        nest.SetKernelStatus({'grng_seed': 984,
                              'rng_seeds': [684]})
        nest.set_verbosity(40)
        graph = nx.DiGraph()
        graph.add_edge(2, 1, weight=1.0)
        graph.add_edge(3, 1, weight=1.0)
        graph.add_edge(4, 1, weight=1.0)

        neuron_pars = {}
        syn_pars = {"model": "static_synapse", "delay": 2.0}
        noise_pars = {"mean": 2.0, "std": 1.0, "dt": 0.1, "frequency": 10.0}
        self.net = neuralnetsim.SubNetwork(graph, 1, neuron_pars, syn_pars,
                                           noise_pars, "iaf_tum_2000", 8000.0,
                                           enable_input_detectors=True)

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

    def test_noise(self):
        nest.SetStatus(self.net._noise, {"mean": 1000.0})
        nest.Simulate(50.0)
        outputs = [5.9, 12.6, 19.3, 26.1, 32.9, 39.7, 46.4]
        for i, spike_time in enumerate(outputs):
            self.assertAlmostEqual(self.net.get_spike_output()[i],
                                   spike_time, 2)

    def test_parrots(self):
        self.net.set_inputs([np.array([10.]), np.array([15.0]), np.array([3.5])])
        nest.Simulate(50.0)
        parrot_output = nest.GetStatus(self.net._parrot_detectors)
        self.assertAlmostEqual(parrot_output[0]['events']['times'][0], 10.1, 1)
        self.assertAlmostEqual(parrot_output[1]['events']['times'][0], 15.1, 1)
        self.assertAlmostEqual(parrot_output[2]['events']['times'][0], 3.6, 1)


class TestOptimizerNetwork(unittest.TestCase):
    def setUp(self) -> None:
        nest.SetKernelStatus({'grng_seed': 16,
                              'rng_seeds': [24]})
        nest.set_verbosity(40)
        graph = nx.DiGraph()
        graph.add_edge(0, 1, weight=1.0)
        graph.add_edge(1, 4, weight=1.0)
        graph.add_edge(4, 0, weight=1.0)
        graph.add_edge(4, 1, weight=1.0)
        self.params = neuralnetsim.CircuitParameters(
            graph,
            "iaf_tum_2000",
            {"t_ref_tot": 10.0, "t_ref_abs": 10.0},
            {"delay": 10.0},
            {"mean": 0.0},
            {"weight_scale": 8000.0}
        )
        self.inputs = {0: np.array([10.0, 100.0, 105.0]),
                       1: np.array([]),
                       4: np.array([200.0])}

    def tearDown(self) -> None:
        nest.ResetKernel()

    def test_setup(self):
        net = neuralnetsim.OptimizerNetwork(self.params, self.inputs)
        self.assertEqual(net._subnets[0].neuron_id, 0)
        self.assertEqual(net._subnets[1].neuron_id, 1)
        self.assertEqual(net._subnets[2].neuron_id, 4)

        self.assertListEqual(net._subnets[0].presynaptic_nodes, [4])
        self.assertListEqual(net._subnets[1].presynaptic_nodes, [0, 4])
        self.assertListEqual(net._subnets[2].presynaptic_nodes, [1])

        self.assertEqual(nest.GetStatus(net._subnets[0]._neuron)[0]['model'],
                         "iaf_tum_2000")
        self.assertAlmostEqual(
            nest.GetStatus(net._subnets[0]._neuron)[0]['t_ref_tot'],
            10.0)
        self.assertAlmostEqual(
            nest.GetStatus(
                nest.GetConnections(net._subnets[0]._parrots,
                                    net._subnets[0]._neuron))[0]['weight'],
            8000.0)

    def test_run(self):
        net = neuralnetsim.OptimizerNetwork(self.params, self.inputs)
        net.run(500.0)
        data = net.get_spike_trains()
        self.assertEqual(list(data.keys()), [0, 1, 4])
        self.assertEqual(len(data[0]), 1)
        self.assertEqual(len(data[4]), 0)
        self.assertEqual(len(data[1]), 3)

        # spikes should occur around 10.0 ms after input due to delay of 10 ms
        self.assertGreater(data[1][0], 20.0)
        self.assertLess(data[1][0], 25.0)
        self.assertGreater(data[1][1], 110.0)
        self.assertLess(data[1][1], 115.0)


class TestTrialManager(unittest.TestCase):
    def setUp(self) -> None:
        nest.set_verbosity(40)
        graph = nx.DiGraph()
        graph.add_edge(0, 1, weight=1.0)
        graph.add_edge(1, 4, weight=1.0)
        graph.add_edge(4, 0, weight=1.0)
        graph.add_edge(4, 1, weight=1.0)
        self.params = neuralnetsim.CircuitParameters(
            graph,
            "iaf_tum_2000",
            {"t_ref_tot": 10.0, "t_ref_abs": 10.0},
            {"delay": 10.0},
            {"mean": 0.0},
            {"weight_scale": 8000.0}
        )
        self.inputs = {0: np.array([10.0, 100.0, 105.0]),
                       1: np.array([]),
                       4: np.array([200.0])}

    def test_context(self):
        run_success = False
        with neuralnetsim.TrialManager(
                {'grng_seed': 16, 'rng_seeds': [24]},
                self.params,
                self.inputs) as net:
            net.run(500.0)
            data = net.get_spike_trains()
            self.assertEqual(list(data.keys()), [0, 1, 4])
            self.assertEqual(len(data[0]), 1)
            self.assertEqual(len(data[4]), 0)
            self.assertEqual(len(data[1]), 3)
            self.assertGreater(data[1][0], 20.0)
            self.assertLess(data[1][0], 25.0)
            self.assertGreater(data[1][1], 110.0)
            self.assertLess(data[1][1], 115.0)
            run_success = True
        self.assertTrue(run_success)
