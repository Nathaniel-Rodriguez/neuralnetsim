import unittest
import neuralnetsim
import numpy as np


class TestCoincidenceFactory(unittest.TestCase):
    def test_coincidence_detector(self):
        model_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.1)
        self.assertEqual(count, 5)

        model_spikes = np.array([0.6, 1.1, 2.1, 3.1, 4.1])
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.2)
        self.assertEqual(count, 5)

        model_spikes = np.array([0.4, 0.9, 1.9, 2.9, 3.9])
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.2)
        self.assertEqual(count, 5)

        model_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        data_spikes = np.array([])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.1)
        self.assertEqual(count, 0)

        model_spikes = np.array([])
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.1)
        self.assertEqual(count, 0)

        model_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        data_spikes = np.array([0.7, 1.5, 2.5, 3.5, 5.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.1)
        self.assertEqual(count, 0)

        model_spikes = np.array([0.9, 1.1, 2.1, 3.1, 3.1])
        data_spikes = np.array([1.0, 2.0, 3.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.2)
        self.assertEqual(count, 5)

        model_spikes = np.array([3.1])
        data_spikes = np.array([1.0, 2.0, 3.0])
        count = neuralnetsim.coincidence_detector(model_spikes, data_spikes, 0.2)
        self.assertEqual(count, 1)

    def test_coincidence_factor(self):
        model_spikes = [np.array([0.5, 1.0, 2.0, 3.0, 4.0])]
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        factor = neuralnetsim.coincidence_factor(model_spikes, data_spikes, 5.0, 0.1)
        self.assertAlmostEqual(factor, -1.0)

        model_spikes = [np.array([0.6, 1.1, 2.1, 3.1, 4.1])]
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        factor = neuralnetsim.coincidence_factor(model_spikes, data_spikes, 5.0, 0.2)
        self.assertAlmostEqual(factor, -1.0)

        model_spikes = [np.array([0.5, 1.0, 2.0, 3.0, 4.0])]
        data_spikes = np.array([])
        factor = neuralnetsim.coincidence_factor(model_spikes, data_spikes, 5.0, 0.1)
        self.assertAlmostEqual(factor, -1.0)

        model_spikes = [np.array([])]
        data_spikes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        factor = neuralnetsim.coincidence_factor(model_spikes, data_spikes, 5.0, 0.1)
        self.assertAlmostEqual(factor, 2.4)

        model_spikes = [np.array([])]
        data_spikes = np.array([])
        factor = neuralnetsim.coincidence_factor(model_spikes, data_spikes, 5.0, 0.2)
        self.assertAlmostEqual(factor, 0.0)
