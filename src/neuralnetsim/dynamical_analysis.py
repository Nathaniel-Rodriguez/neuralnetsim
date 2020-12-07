__all__ = ["bin_spikes",
           "activity",
           "detect_avalanches",
           "isi_distribution_by_neuron",
           "mean_isi"]


import numpy as np
from typing import Dict
from typing import Tuple


def isi_distribution_by_neuron(spike_data: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Returns the inter-spike-interval distribution for each neuron. If there are
    not enough spikes for a given neuron then an empty numpy array is returned
    for that neuron.

    :param spike_data: A dictionary keyed by neuron ID and valued by a numpy
        array of spike times.
    :return: A dictionary keyed by neuron ID and valued by a numpy array of
        inter-spike-intervals.
    """
    return {
        neuron: spikes[1:] - spikes[:-1] if len(spikes) > 1 else np.zeros(0)
        for neuron, spikes in spike_data.items()
    }


def mean_isi(isi_distribution: Dict[int, np.ndarray]) -> float:
    """
    Calculates the mean inter-spike-interval for a given ISI distribution.

    :param isi_distribution: A dictionary keyed by neuron ID and valued by a
        numpy array of inter-spike-intervals.
    :return: The mean ISI.
    """
    return np.mean(np.concatenate(
        list(dist for dist in isi_distribution.values())))


def bin_spikes(spike_data: Dict[int, np.ndarray],
               start_time: float,
               stop_time: float,
               resolution: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts spike data into a binned form where each bin is filled with the
    total number of spikes during that time interval.

    :param spike_data: A dictionary of the spike data for each neuron.
    :param start_time: The start time from which to begin the bins.
    :param stop_time: The end time of the bins.
    :param resolution: The bin size.
    :return: A tuple of the binned data and the bins.
    """
    bins = np.arange(start_time, stop_time + resolution, step=resolution)
    binned = np.zeros(shape=bins.shape, dtype=np.int32)
    mask = np.zeros(shape=bins.shape, dtype=np.int32)
    for spike_train in spike_data.values():
        if len(spike_train) > 0:
            # have to loop in order to count multiple spikes at same index
            for ind in np.digitize(
                spike_train[np.logical_and(
                    spike_train < stop_time,
                    spike_train >= start_time)], bins):
                mask[ind] += 1
            binned += mask  # add spike counts to binned data
            mask[:] = 0  # reset mask
    return binned[1:].copy(), bins


def activity(binned_spikes: np.ndarray,
             filter_size: int) -> np.ndarray:
    """
    Applies a convolution to get the rolling average of the neural activity.

    :param binned_spikes: Spike data binned by aggregated spike count.
    :param filter_size: The size of the box filter.
    :return: Array of averaged activity over time.
    """
    return np.convolve(binned_spikes,
                       np.ones(filter_size) / filter_size,
                       mode='valid')


def detect_avalanches(spike_data: Dict[int, np.ndarray],
                      activity_threshold: float) -> Tuple[int, int]:
    # calculate mean isi
    # choose bin size to match mean isi
    # subtract minimum network activity across whole window (to get rid of const
    # firing neurons) Choose large rolling window size for this
    # When any activity is found, avalanche start and add flag
    # when activity halts, avalanche stop and flip flag
    # choose threshold and only take avalanche size to be that integrated above
    # the threshold
    pass
