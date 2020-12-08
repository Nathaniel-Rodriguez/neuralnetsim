import unittest
import neuralnetsim
import numpy as np


class TestDynamicalAnalysis(unittest.TestCase):
    def test_isi_distribution_by_neuron(self):
        spike_data = {1: np.array([0.0, 1.0, 1.2, 3.0, 5.0]),
                      4: np.array([0.1, 1.2, 3.4, 5.5])}
        dist = neuralnetsim.isi_distribution_by_neuron(spike_data)
        self.assertAlmostEqual(dist[1][0], 1.0)
        self.assertAlmostEqual(dist[1][1], 0.2)
        self.assertAlmostEqual(dist[1][2], 1.8)
        self.assertAlmostEqual(dist[1][3], 2.0)
        self.assertAlmostEqual(dist[4][0], 1.1)
        self.assertAlmostEqual(dist[4][1], 2.2)
        self.assertAlmostEqual(dist[4][2], 2.1)

    def test_mean_isi(self):
        spike_data = {1: np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                      4: np.array([0.1])}
        mean = neuralnetsim.mean_isi(
            neuralnetsim.isi_distribution_by_neuron(
                spike_data
            ))
        self.assertAlmostEqual(mean, 1.0)

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

    def test_detect_avalanches(self):
        binned_activity = np.array([2.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 1.0])
        bins = np.linspace(0.0, 10.0, 11)
        av_times, av_sizes = neuralnetsim.detect_avalanches(
            binned_activity,
            bins,
            0.0
        )
        self.assertEqual(len(av_times), 2)
        self.assertEqual(len(av_sizes), 2)
        self.assertAlmostEqual(av_times[0][0], 0.0)
        self.assertAlmostEqual(av_times[0][1], 2.0)
        self.assertAlmostEqual(av_times[1][0], 4.0)
        self.assertAlmostEqual(av_times[1][1], 5.0)
        self.assertAlmostEqual(av_sizes[0], 3.0)
        self.assertAlmostEqual(av_sizes[1], 1.0)

        binned_activity = np.array([2.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 1.0])
        bins = np.linspace(0.0, 10.0, 11)
        av_times, av_sizes = neuralnetsim.detect_avalanches(
            binned_activity,
            bins,
            1.0
        )
        self.assertEqual(len(av_times), 1)
        self.assertEqual(len(av_sizes), 1)
        self.assertAlmostEqual(av_times[0][0], 0.0)
        self.assertAlmostEqual(av_times[0][1], 1.0)
        self.assertAlmostEqual(av_sizes[0], 1.0)
