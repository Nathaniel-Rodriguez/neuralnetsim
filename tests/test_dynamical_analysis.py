import unittest
import neuralnetsim
import numpy as np


class TestDynamicalAnalysis(unittest.TestCase):
    def test_bin_spikes(self):
        spike_data = {1: np.array([0.0, 1.0, 1.2, 3.0, 5.0]),
                      4: np.array([0.1, 1.2, 3.4, 5.5])}
        binned_spikes, bins = neuralnetsim.bin_spikes(spike_data, 0.0, 4.0)
        self.assertEqual(len(binned_spikes), 4)
        self.assertEqual(len(bins), 5)
        self.assertEqual(binned_spikes[0], 2)
        self.assertEqual(binned_spikes[1], 3)
        self.assertEqual(binned_spikes[2], 0)
        self.assertEqual(binned_spikes[3], 2)

        spike_data = {1: np.array([0.0, 1.0, 1.2, 3.0, 5.0]),
                      4: np.array([])}
        binned_spikes, bins = neuralnetsim.bin_spikes(spike_data, 0.0, 4.0)
        self.assertEqual(len(binned_spikes), 4)
        self.assertEqual(len(bins), 5)
        self.assertEqual(binned_spikes[0], 1)
        self.assertEqual(binned_spikes[1], 2)
        self.assertEqual(binned_spikes[2], 0)
        self.assertEqual(binned_spikes[3], 1)

        spike_data = {1: np.array([]),
                      4: np.array([])}
        binned_spikes, bins = neuralnetsim.bin_spikes(spike_data, 0.0, 4.0)
        self.assertEqual(len(binned_spikes), 4)
        self.assertEqual(len(bins), 5)
        self.assertEqual(binned_spikes[0], 0)
        self.assertEqual(binned_spikes[1], 0)
        self.assertEqual(binned_spikes[2], 0)
        self.assertEqual(binned_spikes[3], 0)

    def test_normalized_activity(self):
        spike_data = {1: np.array([0.0, 1.0, 1.2, 3.0, 5.0]),
                      4: np.array([0.1, 1.2, 3.4, 5.5])}
        binned_spikes, bins = neuralnetsim.bin_spikes(spike_data, 0.0, 5.0, 1)
        norm_act = neuralnetsim.normalized_activity(
            binned_spikes,
            2,
            1
        )
        print(norm_act)

    def test_detect_avalanches(self):
        pass
