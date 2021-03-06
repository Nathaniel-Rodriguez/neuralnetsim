__all__ = ["flow_factor",
           "adjusted_coincidence_factor",
           "coincidence_detector",
           "causal_detector"]


import numpy as np
import bisect
from typing import List


def causal_detector(model_spike_times: np.ndarray,
                    data_spike_times: np.ndarray,
                    coincidence_window: float) -> int:
    """
    Detects the number of spikes in a model time series that occur within a
    given window of spikes in the data time series AFTER a model spike.
    Counts only the first spike that occurs that is closest following the model
    spike.

    :param model_spike_times: A 1-D array of spike times.
    :param data_spike_times: A sorted 1-D array of spike times.
    :param coincidence_window: How close a spike has to be in time to be
        considered coincident.
    :return: Number of coincident spikes.
    """
    coincidence_count = 0
    if len(model_spike_times) == 0:
        return coincidence_count
    if len(data_spike_times) == 0:
        return coincidence_count

    for spike in model_spike_times:
        nearest_right = bisect.bisect_left(data_spike_times, spike)
        if (0 <= (nearest_right - 1)) \
                and ((nearest_right - 1) < (len(data_spike_times) - 1)):
            if 0.0 <= (data_spike_times[nearest_right - 1] - spike) < coincidence_window:
                coincidence_count += 1
                nearest_right += 1
                while ((nearest_right - 1) != len(data_spike_times))\
                        and (0.0 <= (data_spike_times[nearest_right - 1] - spike) < coincidence_window):
                    coincidence_count += 1
                    nearest_right += 1
                continue
        if (0 <= nearest_right) \
                and (nearest_right < (len(data_spike_times) - 1)):
            if 0.0 <= (data_spike_times[nearest_right] - spike) < coincidence_window:
                coincidence_count += 1
                nearest_right += 1
                while (nearest_right != len(data_spike_times))\
                        and (0.0 <= (data_spike_times[nearest_right] - spike) < coincidence_window):
                    coincidence_count += 1
                    nearest_right += 1
                continue
        if nearest_right == (len(data_spike_times) - 1):
            if 0.0 <= (data_spike_times[nearest_right] - spike) < coincidence_window:
                coincidence_count += 1
                continue
        if nearest_right == len(data_spike_times):
            if 0.0 <= (data_spike_times[nearest_right-1] - spike) < coincidence_window:
                coincidence_count += 1
                continue
    return coincidence_count


def coincidence_detector(model_spike_times: np.ndarray,
                         data_spike_times: np.ndarray,
                         coincidence_window: float) -> int:
    """
    Detects the first of spike in a model time series that occur within a
    given window of spikes in the data time series for each corresponding mode
    spike.

    :param model_spike_times: A 1-D array of spike times.
    :param data_spike_times: A 1-D array of spike times.
    :param coincidence_window: How close a spike has to be in time to be
        considered coincident.
    :return: Number of coincident spikes.
    """
    coincidence_count = 0
    if len(model_spike_times) == 0:
        return coincidence_count
    if len(data_spike_times) == 0:
        return coincidence_count

    for spike in model_spike_times:
        nearest_right = bisect.bisect_left(data_spike_times, spike)
        if (0 <= (nearest_right - 1)) \
                and ((nearest_right - 1) < (len(data_spike_times) - 1)):
            if abs(data_spike_times[nearest_right - 1] - spike) < coincidence_window:
                coincidence_count += 1
                continue
        if (0 <= nearest_right) \
                and (nearest_right < (len(data_spike_times) - 1)):
            if abs(data_spike_times[nearest_right] - spike) < coincidence_window:
                coincidence_count += 1
                continue
        if nearest_right == (len(data_spike_times) - 1):
            if abs(data_spike_times[nearest_right] - spike) < coincidence_window:
                coincidence_count += 1
                continue
        if nearest_right == len(data_spike_times):
            if abs(data_spike_times[nearest_right-1] - spike) < coincidence_window:
                coincidence_count += 1
                continue
    return coincidence_count


def adjusted_coincidence_factor(model_spike_times: List[np.ndarray],
                                data_spike_times: np.ndarray,
                                time_window: float,
                                coincidence_window: float = 5.0) -> float:
    """
    Calculates the coincidence factor as described in "Predicting neuronal
    activity with simple models of the threshold type". Adjusted by firing rate
    for use as a cost function.

    :param model_spike_times: A sequence of 1-D arrays with model spike times.
    :param data_spike_times: A 1-D array of data spike times.
    :param time_window: The time window of the experiment.
    :param coincidence_window: How close in time two spikes have to be in order
        to be considered coincident.
    :return: The coincidence factor
    """
    model_num_spikes = np.mean([len(spikes) for spikes in model_spike_times])
    data_num_spikes = len(data_spike_times)
    model_spike_rate = model_num_spikes / time_window
    if data_num_spikes > 0:
        avg_coincidence = np.mean([coincidence_detector(spikes, data_spike_times,
                                                        coincidence_window)
                                   for spikes in model_spike_times])
        data_spike_rate = data_num_spikes / time_window
        expected_coincidence = 2.0 * model_spike_rate * coincidence_window * data_num_spikes
        norm = 2.0 / (1. - 2. * coincidence_window * model_spike_rate)
        cf = 2.0 * abs((data_spike_rate - model_spike_rate) / data_spike_rate) \
                - ((avg_coincidence - expected_coincidence)
                   / (model_num_spikes + data_num_spikes)) * norm
    else:
        cf = -model_spike_rate

    return cf


def flow_factor(model_spike_times: np.ndarray,
                data_spike_times: np.ndarray,
                duration: float,
                coincidence_window: float = 5.0) -> float:
    """
    Calculates the flow factor

    :param model_spike_times: A 1-D arrays with model spike times.
    :param data_spike_times: A 1-D array of data spike times.
    :param duration: The time window of the experiment.
    :param coincidence_window: How long after a spike has to be in order
        to be considered coincident.
    :return: The flow factor
    """
    model_num_spikes = len(model_spike_times)
    data_num_spikes = len(data_spike_times)
    model_spike_rate = model_num_spikes / duration
    if model_num_spikes < 1:
        ff = 0.0
    elif data_num_spikes > 0:
        coincidence = causal_detector(model_spike_times,
                                      data_spike_times,
                                      coincidence_window)
        expected_coincidence = model_spike_rate * coincidence_window * data_num_spikes
        ff = ((coincidence - expected_coincidence)
              / (model_num_spikes + data_num_spikes))
    else:
        ff = 0.0

    return ff
