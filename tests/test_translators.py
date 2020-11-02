import unittest
import neuralnetsim
import networkx as nx
import numpy as np


class TestValueTranslator(unittest.TestCase):
    def test_to_model(self):
        translator = neuralnetsim.ValueTranslator("test", 0, 10)
        self.assertAlmostEqual(translator.to_model(1.0), 10.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_model(0.9), 9.0, 3)
        self.assertAlmostEqual(translator.to_model(0.1), 1.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -10, 10)
        self.assertAlmostEqual(translator.to_model(0.5), 0.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), -10.0, 3)
        self.assertAlmostEqual(translator.to_model(1.0), 10.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -20, -10)
        self.assertAlmostEqual(translator.to_model(0.5), -15.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), -20.0, 3)
        self.assertAlmostEqual(translator.to_model(1.0), -10.0, 3)
        self.assertAlmostEqual(translator.to_model(-0.5), -15.0, 3)
        self.assertAlmostEqual(translator.to_model(2.0), -20.0, 3)

    def test_to_optimizer(self):
        translator = neuralnetsim.ValueTranslator("test", 0, 10)
        self.assertAlmostEqual(translator.to_optimizer(10.0), 1.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(0.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(9.0), 0.9, 3)
        self.assertAlmostEqual(translator.to_optimizer(1.0), 0.1, 3)

        translator = neuralnetsim.ValueTranslator("test", -10, 10)
        self.assertAlmostEqual(translator.to_optimizer(0.0), 0.5, 3)
        self.assertAlmostEqual(translator.to_optimizer(-10.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(10.0), 1.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -20, -10)
        self.assertAlmostEqual(translator.to_optimizer(-15.0), 0.5, 3)
        self.assertAlmostEqual(translator.to_optimizer(-20.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(-10.0), 1.0, 3)

    def test_get_translator(self):
        t1 = neuralnetsim.ValueTranslator("t1", -1, 1)
        t2 = neuralnetsim.ValueTranslator("t2", 0, 10)
        t3 = neuralnetsim.ValueTranslator("tn", 10, 20)
        tlist = [t1, t2, t3]
        self.assertEqual(neuralnetsim.get_translator(tlist, "t1"), t1)
        self.assertRaises(KeyError, neuralnetsim.get_translator, tlist, "t4")


class TestArrayTranslator(unittest.TestCase):
    def setUp(self):
        self.g1 = neuralnetsim.ValueTranslator("g1", 0, 10)
        self.n1 = neuralnetsim.ValueTranslator("n1", -1, 1)
        self.n2 = neuralnetsim.ValueTranslator("n2", 5, 20)
        self.s1 = neuralnetsim.ValueTranslator("s1", 0, 1)
        self.o1 = neuralnetsim.ValueTranslator("o1", -5, -2)
        self.t_list = [self.g1, self.n1, self.n2, self.s1, self.o1]
        self.graph = nx.DiGraph()
        self.graph.add_edge(0, 2)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(3, 0)
        self.g = ["g1"]
        self.n = ["n1", "n2"]
        self.s = ["s1"]
        self.o = ["o1"]
        self.translator = neuralnetsim.ArrayTranslator(
            self.graph, self.t_list, self.n, self.s, self.o, self.g)

    def test_array_size(self):
        self.assertEqual(self.translator.required_array_size(), 3 + 2 * 3)

    def test_from_optimizer(self):
        array = np.array([0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0])
        self.translator.from_optimizer(array)
        self.assertAlmostEqual(self.translator.global_parameters['g1'], 0.0)
        self.assertAlmostEqual(self.translator.noise_parameters['o1'], -2.0)
        self.assertAlmostEqual(self.translator.synapse_parameters['s1'], 0.5)
        self.assertAlmostEqual(self.translator.neuron_parameters[0]['n1'], -1.0)
        self.assertAlmostEqual(self.translator.neuron_parameters[0]['n2'], 20.0)
        self.assertAlmostEqual(self.translator.neuron_parameters[2]['n1'], 1.0)
        self.assertAlmostEqual(self.translator.neuron_parameters[2]['n2'], 5.0)
        self.assertAlmostEqual(self.translator.neuron_parameters[3]['n1'], 0.0)
        self.assertAlmostEqual(self.translator.neuron_parameters[3]['n2'], 20.0)

    def test_to_optimizer(self):
        array = np.array([0.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0])
        self.translator.from_optimizer(array)
        opt_array = self.translator.to_optimizer()
        for i in range(len(array)):
            self.assertAlmostEqual(array[i], opt_array[i])
