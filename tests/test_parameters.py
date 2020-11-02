import networkx as nx
import unittest
import neuralnetsim


class TestCircuitParameters(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_edge(0, 1)
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 0)

    def test_setup(self):
        pars = neuralnetsim.CircuitParameters(
            self.graph, "toast", {'n1': 1.0}, {'s1': 2.0}, {'o1': 0.1},
            {'g1': 0.0})
        self.assertDictEqual(pars.neuron_parameters,
                             {0: {'n1': 1.0}, 1: {'n1': 1.0}, 2: {'n1': 1.0}})
        self.assertDictEqual(pars.synaptic_parameters, {'s1': 2.0})
        self.assertDictEqual(pars.noise_parameters, {'o1': 0.1})
        self.assertDictEqual(pars.global_parameters, {'g1': 0.0})
        self.assertDictEqual(pars.kernel_parameters, dict())

    def test_extend_parameters(self):
        pars = neuralnetsim.CircuitParameters(
            self.graph, "toast", {'n1': 1.0}, {'s1': 2.0}, {'o1': 0.1},
            {'g1': 0.0})
        pars.extend_global_parameters({'g2': 2.0})
        pars.extend_noise_parameters({'o2': 3.0})
        pars.extend_synaptic_parameters({'s2': 4.0})
        pars.extend_neuron_parameters({0: {'n2': 0.0}, 1: {'n2': 0.0}, 2: {'n2': 0.0}})

        self.assertDictEqual(pars.neuron_parameters,
                             {0: {'n1': 1.0, 'n2': 0.0},
                              1: {'n1': 1.0, 'n2': 0.0},
                              2: {'n1': 1.0, 'n2': 0.0}})
        self.assertDictEqual(pars.synaptic_parameters, {'s1': 2.0, 's2': 4.0})
        self.assertDictEqual(pars.noise_parameters, {'o1': 0.1, 'o2': 3.0})
        self.assertDictEqual(pars.global_parameters, {'g1': 0.0, 'g2': 2.0})
