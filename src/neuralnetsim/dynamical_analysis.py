__all__ = ["bin_spikes",
           "activity",
           "normalized_activity",
           "detect_avalanches"]


import numpy as np
from typing import Dict
from typing import Tuple


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


def normalized_activity(circuit_activity: np.ndarray,
                        num_neurons: int,
                        resolution: float) -> np.ndarray:
    """
    Normalizes the circuit activity between 0 and 1.

    :param circuit_activity: The neural network activity
    :param num_neurons: The number of neurons used to calculate the activity.
    :param resolution: The size of the bins in time.
    :return: Array of normalized activity.
    """
    return circuit_activity / num_neurons / resolution


def detect_avalanches(spike_data: Dict[int, np.ndarray],
                      activity_threshold: float) -> Tuple[int, int]:
    # maybe take a different approach than activity, which seems resolution dep
    # just needs start and end times of transition
    # then participation and fire counts within that time window can be
    # aggregated. So just need a transition detection algorithm that doesnt
    # require a time resolution or threshold
    pass
