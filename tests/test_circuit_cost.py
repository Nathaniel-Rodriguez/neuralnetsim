import unittest
import neuralnetsim
import numpy as np
import networkx as nx
import nest


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
            {},
            {"model": "static_synapse", "delay": 1.0},
            {"mean": 2.0, "std": 1.0, "dt": 0.1, "frequency": 10.0}
        )
        kpars = {'grng_seed': 684, 'rng_seeds': [658], 'resolution': 0.1}
        trans = neuralnetsim.ArrayTranslator(
            graph,
            [neuralnetsim.ValueTranslator("delay", 0.0, 2.0),
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
        print(cost)
