import unittest
import neuralnetsim
import numpy as np
import networkx as nx


class TestCricuitCost(unittest.TestCase):
    def test_circuit_cost(self):
        spike_trains = {0: np.array([0.0, 5.0, 10.0, 15.0]),
                        2: np.array([1.1, 6.1, 11.1, 16.1]),
                        4: np.array([1.0, 5.0]),
                        5: np.array([2.0, 6.0]),
                        6: np.array([])}
        dm = neuralnetsim.DataManager(spike_trains, 1, start_buffer=0.1)
        graph = nx.DiGraph()
        graph.add_edge(0, 2, weight=1.0)
        graph.add_edge(4, 5, weight=1.0)
        graph.add_edge(6, 5, weight=1.0)
        pars = neuralnetsim.CircuitParameters(
            graph, "iaf_tum_2000",
            {'t_ref_abs': 1.0, 't_ref_tot': 1.0},
            {"model": "static_synapse", "delay": 0.1},
            {"mean": 2.0, "std": 1.0, "dt": 0.1, "frequency": 10.0}
        )
        kpars = {'grng_seed': 684, 'rng_seeds': [658], 'resolution': 0.1}
        trans = neuralnetsim.ArrayTranslator(
            graph,
            [neuralnetsim.ValueTranslator("delay", 0.0, 5.0),
             neuralnetsim.ValueTranslator("weight_scale", 0.0, 8000.0)],
            ["weight_scale"],
            [],
            ["delay"],
            []
        )
        cost = neuralnetsim.circuit_cost(
            np.array([1.0, 0.5]),
            pars,
            trans,
            kpars,
            dm.get_training_fold(0),
            dm.get_duration("training", 0, 2.0),
            5.0
        )
        self.assertAlmostEqual(13.99, cost, 1)


class TestIzhikevichMemberSource(unittest.TestCase):
    def test_draw_method(self):
        translators = [
            neuralnetsim.ValueTranslator(
                'a', 0.0, 0.04),  # 0.5
            neuralnetsim.ValueTranslator(
                'b', 0.2, 0.4),  # 0.0
            neuralnetsim.ValueTranslator(
                'c', -75.0, -65.0),  # 1.0
            neuralnetsim.ValueTranslator(
                'd', 0.0, 32.0),  # 0.25
            neuralnetsim.ValueTranslator(
                'mode', 0.0, 1.0),  # either 0.25 or 0.75
            neuralnetsim.DistributionTranslator(
                'delay', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)}, 0.4, 8.0),
            neuralnetsim.DistributionTranslator(
                'U', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)}),
            neuralnetsim.DistributionTranslator(
                'u', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)}),
            neuralnetsim.DistributionTranslator(
                'x', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)},
                0.0, 10.),
            neuralnetsim.DistributionTranslator(
                'tau_rec', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)},
                0.04, 3000.0),
            neuralnetsim.DistributionTranslator(
                'tau_fac', "beta",
                {'a': (0.01, 100),
                 'b': (0.01, 100)},
                0.04, 2000.0),
            neuralnetsim.ValueTranslator('rate', 0.0, 3000.0, False),
            neuralnetsim.ValueTranslator('weight_scale', 4e5, 6e7, True)
        ]
        graph = nx.DiGraph()
        graph.add_edge(0, 2, weight=1.0)
        graph.add_edge(4, 5, weight=1.0)
        graph.add_edge(6, 5, weight=1.0)
        parameters = neuralnetsim.AllNeuronSynDistParameters(
            graph, "izhikevich",
            translators,
            ["weight_scale"], ['rate'],
            ["U", "u", "x", "tau_rec", "tau_fac", "delay"],
            ["a", "b", "c", "d", "mode"],
            {}, {}, {}, {}
        )

        ms = neuralnetsim.IzhikevichMemberSource(
            parameters
        )
        rng = np.random.RandomState(seed=1)
        test_draw = ms.draw(rng)
        # end 13
        self.assertAlmostEqual(test_draw[14], 0.5)
        self.assertAlmostEqual(test_draw[15], 0.0)
        self.assertAlmostEqual(test_draw[16], 1.0)
        self.assertAlmostEqual(test_draw[17], 0.25)
        if test_draw[18] < 0.5:
            self.assertAlmostEqual(test_draw[18], 0.25)
        elif test_draw[18] > 0.5:
            self.assertAlmostEqual(test_draw[18], 0.75)

        self.assertAlmostEqual(test_draw[19], 0.5)
        self.assertAlmostEqual(test_draw[20], 0.0)
        self.assertAlmostEqual(test_draw[21], 1.0)
        self.assertAlmostEqual(test_draw[22], 0.25)
        if test_draw[23] < 0.5:
            self.assertAlmostEqual(test_draw[23], 0.25)
        elif test_draw[23] > 0.5:
            self.assertAlmostEqual(test_draw[23], 0.75)

        self.assertAlmostEqual(test_draw[24], 0.5)
        self.assertAlmostEqual(test_draw[25], 0.0)
        self.assertAlmostEqual(test_draw[26], 1.0)
        self.assertAlmostEqual(test_draw[27], 0.25)
        if test_draw[28] < 0.5:
            self.assertAlmostEqual(test_draw[28], 0.25)
        elif test_draw[28] > 0.5:
            self.assertAlmostEqual(test_draw[28], 0.75)
