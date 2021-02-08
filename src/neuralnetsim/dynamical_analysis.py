__all__ = ["spike_count",
           "bin_spikes",
           "activity",
           "detect_avalanches",
           "network_isi_distribution",
           "isi_distribution_by_neuron",
           "mean_isi",
           "avalanches_from_zero_activity",
           "participating_neuron_distribution",
           "get_acorr_time",
           "embed_time_series",
           "generate_umap_map",
           "generate_persistence_diagrams",
           "diagram_distances",
           "firing_rate_distribution",
           "process_sim_results",
           "agg_sim_avalanche_distributions",
           "effective_flow",
           "bridge_flows",
           "process_bridge_results",
           "percent_active",
           "community_outflow",
           "internal_community_flow",
           "process_outflow_results",
           "neighbor_flow",
           "global_flow",
           "process_global_flow_results",
           "process_internal_flow_results",
           "process_grid_results",
           "find_bridges"]


import neuralnetsim
import networkx as nx
import numpy as np
import umap
import ripser
import statsmodels.tsa.stattools as stats
import pandas as pd
from distributed import Client
from collections import Counter
from pathlib import Path
from persim import sliced_wasserstein
from typing import Dict
from typing import Tuple
from typing import List


def percent_active(data: Dict[int, np.ndarray]) -> float:
    return sum([1 for arr in data.values() if len(arr) > 1]) / len(data)


def spike_count(data: Dict[int, np.ndarray]) -> int:
    """
    Count the number of total spikes emitted by the neural network in the data.

    :param data: A dictionary keyed by neuron ID and valued by a numpy
        array of spike times.
    :return: The number of spikes.
    """
    num_spikes = 0
    for arr in data.values():
        num_spikes += len(arr)
    return num_spikes


def effective_flow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        neuron_id: int,
        comm_id: int,
        coincidence_window: float,
        duration: float,
        com_key: str
) -> float:
    """

    :param data:
    :param graph:
    :param neuron_id:
    :param comm_id:
    :param coincidence_window:
    :param duration:
    :param com_key:
    :return:
    """
    down_stream_neurons = list(nx.neighbors(graph, neuron_id))
    down_stream_neurons_in_com = [
        neuron for neuron in down_stream_neurons
        if graph.nodes[neuron][com_key] == comm_id
    ]
    com_spikes = np.array(
        [spike_time
         for neuron in down_stream_neurons_in_com
         for spike_time in data[neuron]]
    )
    com_spikes.sort()

    ef = neuralnetsim.flow_factor(
        data[neuron_id], com_spikes, duration, coincidence_window)

    return ef


def internal_flow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        neuron_id: int,
        coincidence_window: float,
        duration: float,
        com_key: str
):
    down_stream_neurons = list(nx.neighbors(graph, neuron_id))
    down_stream_neurons_own_com = [
        neuron for neuron in down_stream_neurons
        if graph.nodes[neuron][com_key] == graph.nodes[neuron_id][com_key]
    ]
    com_spikes = np.array(
        [spike_time
         for neuron in down_stream_neurons_own_com
         for spike_time in data[neuron]]
    )
    com_spikes.sort()

    ef = neuralnetsim.flow_factor(
        data[neuron_id], com_spikes, duration, coincidence_window)

    return ef


def neighbor_flow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        neuron_id: int,
        coincidence_window: float,
        duration: float,
):
    down_stream_neurons = list(nx.neighbors(graph, neuron_id))
    com_spikes = np.array(
        [spike_time
         for neuron in down_stream_neurons
         for spike_time in data[neuron]]
    )
    com_spikes.sort()
    ef = neuralnetsim.flow_factor(
        data[neuron_id], com_spikes, duration, coincidence_window)

    return ef


def find_bridges(
        graph: nx.DiGraph,
        com_key: str
) -> List[Tuple[int, int]]:
    """

    :param graph:
    :param com_key:
    :return: List of tuples of bridge neurons with the corresponding community
        they are a bridge for. Neurons can appear multiple times if they are
        bridges for multiple communities. First index is neuron id, second
        index of tuple is community id.
    """
    bridge_nodes = []
    for node_i in graph.nodes:
        neighbors = nx.neighbors(graph, node_i)
        for neighbor in neighbors:
            if graph.nodes[node_i][com_key] != graph.nodes[neighbor][com_key]:
                bridge_nodes.append((node_i, graph.nodes[neighbor][com_key]))
    return bridge_nodes


def bridge_flows(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        coincidence_window: float,
        duration: float,
        com_key: str
) -> np.ndarray:
    """

    :param data:
    :param graph:
    :param coincidence_window:
    :param duration:
    :param com_key:
    :return:
    """
    bridge_nodes = find_bridges(graph, com_key)
    flows = []
    for bridge, com in bridge_nodes:
        flows.append(effective_flow(
            data, graph, bridge, com, coincidence_window, duration, com_key))
    return np.array(flows)


def community_outflow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        coincidence_window: float,
        duration: float,
        com_key: str
) -> Dict[int, np.ndarray]:
    """
    Calculates bridge out-flow by community

    :param data:
    :param graph:
    :param coincidence_window:
    :param duration:
    :param com_key:
    :return:
    """
    bridge_nodes = find_bridges(graph, com_key)
    coms = neuralnetsim.get_communities(graph, com_key)
    return {
        com: np.array([
            effective_flow(data, graph, bridge,
                           sink_com, coincidence_window,
                           duration, com_key)
            for bridge, sink_com in bridge_nodes
            if graph.nodes[bridge][com_key] == com
        ])
        for com in coms
    }


def internal_community_flow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        coincidence_window: float,
        duration: float,
        com_key: str
) -> Dict[int, np.ndarray]:
    coms = neuralnetsim.get_communities(graph, com_key)
    return {
        com: np.array([
            internal_flow(data, graph, node, coincidence_window,
                          duration, com_key)
            for node in graph.nodes
            if graph.nodes[node][com_key] == com
        ])
        for com in coms
    }


def global_flow(
        data: Dict[int, np.ndarray],
        graph: nx.DiGraph,
        coincidence_window: float,
        duration: float
) -> np.ndarray:
    return np.array([
            neighbor_flow(data, graph, node, coincidence_window,
                          duration)
            for node in graph.nodes
        ])


def firing_rate_distribution(data: Dict[int, np.ndarray],
                             duration: float) -> np.ndarray:
    firing_rates = []
    for spikes in data.values():
        firing_rates.append(len(spikes) / duration)
    return np.array(firing_rates)


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


def network_isi_distribution(spike_data: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Calculates the ISI distribution of the network as a whole. This is the
    interspike-interval between any spikes or any neurons in the network.

    :param spike_data: A dictionary keyed by neuron ID and valued by a numpy
        array of spike times.
    :return: An array of the ISI distribution.
    """
    dist = np.concatenate(list(times for times in spike_data.values()))
    dist.sort()
    return dist[1:] - dist[:-1]


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


def detect_avalanches(
        binned_spikes: np.ndarray,
        bins: np.ndarray,
        activity_threshold: float,
        discrete_times: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects avalanches given spike data from the neural network. Cuts off the
    last avalanche if it does not end before the last bin, since it is not
    possible to determine its duration.

    :param binned_spikes: The aggregated spiking data of the neural network.
    :param bins: The time values associated with each bin.
    :param activity_threshold: The activity level above which to consider an
        avalanche to have begun or ended.
    :return: A 2-D numpy array of size Kx2 where K is the number of avalanches.
        The first dimension is the avalanche start time, and the second is the
        avalanche end time. The second 1-D K size array gives the size of the
        corresponding avalanches. The size is the total number of spikes
        above the baseline threshold integrated between the start and end times.
    """
    bin_size = bins[1] - bins[0]
    avalanche_mask = binned_spikes > activity_threshold
    avalanche = False
    avalanche_start = 0
    avalanche_end = avalanche_start
    avalanche_size = 0
    avalanche_times = []
    avalanche_sizes = []
    for i in range(len(binned_spikes)):
        if not avalanche:
            if avalanche_mask[i]:
                avalanche = True
                avalanche_size = binned_spikes[i]
                if discrete_times:
                    avalanche_start = i
                else:
                    avalanche_start = bins[i]
        else:
            if avalanche_mask[i]:
                avalanche_size += binned_spikes[i]
            else:
                if discrete_times:
                    avalanche_end = i
                else:
                    avalanche_end = bins[i]
                avalanche = False
                avalanche_times.append((avalanche_start, avalanche_end))
                avalanche_sizes.append(
                    avalanche_size * bin_size
                    - activity_threshold * (avalanche_end - avalanche_start))
                avalanche_size = 0.0

    return np.array(avalanche_times), np.array(avalanche_sizes)


def avalanches_from_zero_activity(
        spike_data: Dict[int, np.ndarray],
        start_time: float,
        stop_time: float,
        resolution: float = None,
        discrete_times: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the avalanches based on the zero level of network activity
    based on binned total activity. Bin size is the mean network ISI.

    :param spike_data: A dictionary of the spike data for each neuron.
    :param start_time: The start time from which to begin the bins.
    :param stop_time: The end time of the bins.
    :param resolution: Defaults to network ISI. This is the bin size.
    :return: A 2-D numpy array of size Kx2 where K is the number of avalanches.
        The first dimension is the avalanche start time, and the second is the
        avalanche end time. The second 1-D K size array gives the size of the
        corresponding avalanches. The size is the total number of spikes
        above the baseline threshold integrated between the start and end times.
    """
    if resolution is None:
        resolution = np.mean(network_isi_distribution(
            {node: values[np.logical_and(values >= start_time,
                                         values < stop_time)]
             for node, values in spike_data.items()}
        ))
        if np.isnan(resolution):
            print(start_time, stop_time,
                  sum(len(s) for s in spike_data.values()), flush=True)
            raise ValueError("NAN RESOLUTION: FAILED TO CALCULATE ISI")
    spikes, bins = bin_spikes(spike_data, start_time, stop_time, resolution)
    return detect_avalanches(
        spikes,
        bins,
        0.0,
        discrete_times
    )


# def participating_neuron_distribution(avalanche_times: np.ndarray,
#                                       spike_data: Dict[int, np.ndarray]) -> np.ndarray:
#     """
#
#     :param avalanche_times:
#     :param spike_data:
#     :return:
#     """
#     import time
#     s = time.time()
#     participation_dist = np.zeros(len(avalanche_times), dtype=np.int32)
#     for i, (start, stop) in enumerate(avalanche_times):
#         for spikes in spike_data.values():
#             if len(spikes[np.logical_and(
#                 spikes >= start,
#                 spikes < stop
#             )]) >= 1:
#                 participation_dist[i] += 1
#     print("part time", time.time() - s, "num aval", len(avalanche_times), flush=True)
#     return participation_dist


def participating_neuron_distribution(
        avalanche_times: np.ndarray,
        spike_data: Dict[int, np.ndarray]
) -> np.ndarray:
    bins = np.sort(avalanche_times.flatten())
    total_participation = []
    for spikes in spike_data.values():
        total_participation += list(set(np.digitize(spikes, bins)))
    return np.array(list(Counter(total_participation).values()))


def get_acorr_time(binned_spikes: np.ndarray, lags=1000, threshold=0.5) -> int:
    acorr = stats.acf(binned_spikes, nlags=lags, fft=True)
    acorr_time = 0
    while (acorr_time < len(acorr)) and (acorr[acorr_time] > threshold):
        acorr_time += 1
    return acorr_time


def embed_time_series(x: np.ndarray,
                      tau: int,
                      dimensions: int) -> np.ndarray:
    """
    :param x: 1-D time-series (length N)
    :param tau: number of time-steps back for delay
    :param dimensions: dimensionality of the embedding
    :return: (N - dimensions*tau) x (dimensions)
    """
    if len(x) < dimensions * tau:
        raise ValueError(
            "Time series can't accommodate embedding: "
            + "Num points {0}, tau {1}, required {2}".format(
                str(len(x)), str(tau), str(tau * dimensions))
        )
    embedding = np.zeros(shape=(len(x) - dimensions * tau, dimensions))
    for dim in range(embedding.shape[1]):
        embedding[:, dim] = x[dimensions * tau - dim * tau:len(x) - dim * tau]

    return embedding


def generate_umap_map(
        binned_data,
        dimensionality,
        max_lags,
        **kwargs) -> umap.UMAP:
    lag_time = get_acorr_time(binned_data, max_lags)
    embeding = embed_time_series(binned_data, lag_time, dimensionality)
    fit = umap.UMAP(
        **kwargs
    )
    return fit.fit(embeding)


def generate_persistence_diagrams(
        binned_data: np.ndarray,
        dimensionality: int,
        max_lags,
        **kwargs
) -> List[np.ndarray]:
    """

    :param binned_data:
    :param dimensionality:
    :param kwargs:
    :return:
    """
    ripargs = kwargs.copy()
    lag_time = get_acorr_time(binned_data, max_lags)
    print("lag time", lag_time)
    embeding = embed_time_series(binned_data, lag_time, dimensionality)
    if (ripargs["n_perm"] is not None) \
            and (embeding.shape[0] < ripargs["n_perm"]):
        ripargs["n_perm"] = None
    return ripser.ripser(embeding, **ripargs)['dgms']


def diagram_distances(
        diagrams1,
        diagrams2
) -> float:
    # assume first in each list is H0, drop inf
    assert(len(diagrams1) == len(diagrams2))
    total_distance = 0
    for i in range(len(diagrams1)):
        if i == 0:
            d1 = diagrams1[i][~np.isinf(diagrams1[i]).any(axis=1)]
            d2 = diagrams2[i][~np.isinf(diagrams2[i]).any(axis=1)]
            total_distance += sliced_wasserstein(d1, d2)
        else:
            total_distance += sliced_wasserstein(diagrams1[i], diagrams2[i])
    return total_distance


def largest_community(
        node_communities: Dict[int, int],
        nth_largest: int
):
    return Counter(node_communities.values()).most_common(nth_largest)


def bridge_worker(
        graph,
        data,
        par,
        mu,
        duration,
        causal_window,
        com_key
):
    bf = np.mean(neuralnetsim.bridge_flows(data, graph, causal_window, duration, com_key))
    gf = np.mean(neuralnetsim.global_flow(data, graph, causal_window, duration))
    cf = np.mean(np.concatenate([efs for efs in neuralnetsim.internal_community_flow(
            data, graph, causal_window, duration, com_key).values()]))
    return {'max duration': duration,
            r'$\mu$': mu,
            'flow': bf,
            'gflow': gf,
            'cflow': cf,
            'grid_par': par,
            'activity': neuralnetsim.spike_count(data) / duration / nx.number_of_nodes(graph)
            }


def process_bridge_results(
        sim_path: Path,
        client: Client,
        causal_window: float,
        com_key: str
) -> pd.DataFrame:
    sim_data = neuralnetsim.load(sim_path)
    results = client.map(
        bridge_worker,
        [graph for graph in sim_data['graphs']],
        [data for data in sim_data['spike_data']],
        [par for par in sim_data['grid_par']],
        [mu for mu in sim_data['target_modularities']],
        causal_window=causal_window,
        duration=sim_data['duration'],
        com_key=com_key,
        pure=False
    )
    return pd.DataFrame(client.gather(results))


def outflow_worker(
        data,
        control_var,
        duration,
        graph,
        causal_window,
        com_key
) -> pd.DataFrame:
    outflow = neuralnetsim.community_outflow(
        data,
        graph,
        causal_window,
        duration,
        com_key)
    cflow = neuralnetsim.internal_community_flow(
        data, graph, causal_window, duration, com_key)
    return pd.DataFrame(
        [{
            'activity': neuralnetsim.spike_count(
                neuralnetsim.community_sub_data(
                    com, data, graph, com_key))
                        / duration
                        / neuralnetsim.get_community_size(
                graph, com_key, com),
            'flow': np.mean(flows),
            'cflow': np.mean(cflow[com]),
            'com': com,
            'control_var': control_var
        } for com, flows in outflow.items()]
    )


def process_outflow_results(
        sim_path: Path,
        graph: nx.DiGraph,
        client: Client,
        causal_window: float,
        com_key: str
) -> pd.DataFrame:
    sim_data = neuralnetsim.load(sim_path)
    results = client.map(
        outflow_worker,
        [data for data in sim_data['spike_data']],
        [par for par in sim_data['control_var']],
        causal_window=causal_window,
        duration=sim_data['duration'],
        graph=graph,
        com_key=com_key,
        pure=False
    )
    return pd.concat(client.gather(results), ignore_index=True)


def global_flow_worker(
        data,
        control_var,
        duration,
        graph,
        causal_window
) -> pd.DataFrame:
    global_flows = neuralnetsim.global_flow(
        data,
        graph,
        causal_window,
        duration)
    return pd.DataFrame(
        [{
            'flow': flow,
            'control_var': control_var
        } for flow in global_flows]
    )


def process_global_flow_results(
        sim_path: Path,
        graph: nx.DiGraph,
        client: Client,
        causal_window: float
) -> pd.DataFrame:
    sim_data = neuralnetsim.load(sim_path)
    results = client.map(
        global_flow_worker,
        [data for data in sim_data['spike_data']],
        [par for par in sim_data['control_var']],
        causal_window=causal_window,
        duration=sim_data['duration'],
        graph=graph,
        pure=False
    )
    return pd.concat(client.gather(results), ignore_index=True)


def internal_flow_worker(
        data,
        control_var,
        duration,
        graph,
        causal_window,
        com_key
) -> pd.DataFrame:
    outflow = neuralnetsim.internal_community_flow(
        data,
        graph,
        causal_window,
        duration,
        com_key)
    return pd.DataFrame(
        [{
            'flow': np.mean(flows),
            'com': com,
            'control_var': control_var
        } for com, flows in outflow.items()]
    )


def process_internal_flow_results(
        sim_path: Path,
        graph: nx.DiGraph,
        client: Client,
        causal_window: float,
        com_key: str
) -> pd.DataFrame:
    sim_data = neuralnetsim.load(sim_path)
    results = client.map(
        internal_flow_worker,
        [data for data in sim_data['spike_data']],
        [par for par in sim_data['control_var']],
        causal_window=causal_window,
        duration=sim_data['duration'],
        graph=graph,
        com_key=com_key,
        pure=False
    )
    return pd.concat(client.gather(results), ignore_index=True)


def grid_worker(
        graph,
        data,
        par,
        mu,
        duration,
        causal_window,
        com_key
):
    cf = neuralnetsim.internal_community_flow(
            data,
            graph,
            causal_window,
            duration,
            com_key)
    gf = neuralnetsim.global_flow(
        data,
        graph,
        causal_window,
        duration
    )
    return (pd.DataFrame([{
        r'$\mu$': mu,
        'flow': np.mean(flows),
        'com': com,
        'control_var': par,
    } for com, flows in cf.items()]),
        pd.DataFrame(
            [{
                r'$\mu$': mu,
                'flow': flow,
                'control_var': par
            } for flow in gf]
        )
    )


def process_grid_results(
        sim_path: Path,
        client: Client,
        causal_window: float,
        com_key: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sim_data = neuralnetsim.load(sim_path)
    results = client.map(
        grid_worker,
        [graph for graph in sim_data['graphs']],
        [data for data in sim_data['spike_data']],
        [par for par in sim_data['grid_par']],
        [mu for mu in sim_data['target_modularities']],
        causal_window=causal_window,
        duration=sim_data['duration'],
        com_key=com_key,
        pure=False
    )
    com_results, global_results = zip(*client.gather(results))
    return pd.concat(com_results, ignore_index=True),\
           pd.concat(global_results, ignore_index=True)


def process_sim_results(
        graphs_path: Path,
        sim_path: Path
) -> pd.DataFrame:
    ######### need to make 1 entry per DATA POINT, not 1 per set of datapoints
    ##### coms values need their own data points
    ### slope stuff should be in their own pd
    graph_data = neuralnetsim.load(graphs_path)
    sim_data = neuralnetsim.load(sim_path)
    assert len(graph_data['graphs']) == len(sim_data['spike_data'])
    av_size_slope = []
    av_truncated_size_slope = []
    av_duration_slope = []
    av_truncated_duration_slope = []
    av_mean_sizes = []
    av_mean_durations = []
    av_com1_mean_sizes = []
    av_com1_mean_durations = []
    av_com2_mean_sizes = []
    av_com2_mean_durations = []
    av_com3_mean_sizes = []
    av_com3_mean_durations = []
    durations = []
    mus = []
    trials = []
    for i in range(len(graph_data['graphs'])):
        mus.append(graph_data['target_modularities'][i])
        trials.append(graph_data['trials'][i])
        durations.append(sim_data['duration'])

        if spike_count(sim_data['spike_data'][i]) > 10:
            # network wide stats
            bins_size = float(np.mean(network_isi_distribution(sim_data['spike_data'][i])))
            av_times, av_sizes = avalanches_from_zero_activity(
                sim_data['spike_data'][i],
                0.0,
                sim_data['duration'],
                bins_size,
                True
            )
            av_durations = av_times[:, 1] - av_times[:, 0]
            # sfit = powerlaw.Fit(av_sizes, discrete=True)
            # av_size_slope.append(sfit.alpha)
            # av_truncated_size_slope.append(sfit.truncated_power_law.parameter1)
            # dfit = powerlaw.Fit(av_durations, discrete=True)
            # av_duration_slope.append(dfit.alpha)
            # av_truncated_duration_slope.append(dfit.truncated_power_law.parameter1)
            av_mean_sizes.append(np.mean(av_sizes))
            av_mean_durations.append(np.mean(av_durations) * bins_size)

            # collected community data stats
            node_coms = nx.get_node_attributes(graph_data['graphs'][i], "level1")
            largest_coms = largest_community(node_coms, 3)
            com1_spikes = {node: sim_data['spike_data'][i][node]
                           for node, com in node_coms.items()
                           if com == largest_coms[0][0]}
            if len(largest_coms) > 1:
                com2_spikes = {node: sim_data['spike_data'][i][node]
                               for node, com in node_coms.items()
                               if com == largest_coms[1][0]}
            else:
                com2_spikes = {}
            if len(largest_coms) > 2:
                com3_spikes = {node: sim_data['spike_data'][i][node]
                               for node, com in node_coms.items()
                               if com == largest_coms[2][0]}
            else:
                com3_spikes = {}

            # community stats
            if len(com1_spikes) >= 4 and spike_count(com1_spikes) > 10:
                bins_size = float(np.mean(network_isi_distribution(com1_spikes)))
                av_times, av_sizes = avalanches_from_zero_activity(
                    com1_spikes,
                    0.0,
                    sim_data['duration'],
                    bins_size,
                    False
                )
                av_durations = av_times[:, 1] - av_times[:, 0]
                av_com1_mean_sizes.append(np.mean(av_sizes))
                av_com1_mean_durations.append(np.mean(av_durations))
            else:
                av_com1_mean_sizes.append(0.0)
                av_com1_mean_durations.append(0.0)

            if len(com2_spikes) >= 4 and spike_count(com2_spikes) > 10:
                bins_size = float(np.mean(network_isi_distribution(com2_spikes)))
                av_times, av_sizes = avalanches_from_zero_activity(
                    com2_spikes,
                    0.0,
                    sim_data['duration'],
                    bins_size,
                    False
                )
                av_durations = av_times[:, 1] - av_times[:, 0]
                av_com2_mean_sizes.append(np.mean(av_sizes))
                av_com2_mean_durations.append(np.mean(av_durations))
            else:
                av_com2_mean_sizes.append(0.0)
                av_com2_mean_durations.append(0.0)

            if len(com3_spikes) >= 4 and spike_count(com3_spikes) > 10:
                bins_size = float(np.mean(network_isi_distribution(com3_spikes)))
                av_times, av_sizes = avalanches_from_zero_activity(
                    com3_spikes,
                    0.0,
                    sim_data['duration'],
                    bins_size,
                    False
                )
                av_durations = av_times[:, 1] - av_times[:, 0]
                av_com3_mean_sizes.append(np.mean(av_sizes))
                av_com3_mean_durations.append(np.mean(av_durations))
            else:
                av_com3_mean_sizes.append(0.0)
                av_com3_mean_durations.append(0.0)
        else:
            av_mean_sizes.append(0.0)
            av_mean_durations.append(0.0)
            av_com1_mean_sizes.append(0.0)
            av_com1_mean_durations.append(0.0)
            av_com2_mean_sizes.append(0.0)
            av_com2_mean_durations.append(0.0)
            av_com3_mean_sizes.append(0.0)
            av_com3_mean_durations.append(0.0)

    return pd.DataFrame({
        # 'size slope': av_size_slope,
        # 'truncated size slope': av_truncated_size_slope,
        # 'duration slope': av_duration_slope,
        # 'truncated duration slope': av_truncated_duration_slope,
        'size': av_mean_sizes,
        'duration': av_mean_durations,
        'com 1 size': av_com1_mean_sizes,
        'com 1 duration': av_com1_mean_durations,
        'com 2 size': av_com2_mean_sizes,
        'com 2 duration': av_com2_mean_durations,
        'com 3 sizes': av_com3_mean_sizes,
        'com 3 duration': av_com3_mean_durations,
        'max duration': durations,
        r'$\mu$': mus,
        'trial': trials
    })


def agg_sim_avalanche_distributions(
        graphs_path: Path,
        sim_path: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    graph_data = neuralnetsim.load(graphs_path)
    sim_data = neuralnetsim.load(sim_path)
    assert len(graph_data['graphs']) == len(sim_data['spike_data'])

    mus = graph_data['parameters']['modularities']
    sizes = [np.zeros(0) for _ in mus]
    durations = [np.zeros(0) for _ in mus]
    for i, mu in enumerate(mus):
        for j in range(len(graph_data['graphs'])):
            if graph_data['target_modularities'][j] == mu:
                # no meaningful isi if circuit not active (10 is arbitrary)
                if spike_count(sim_data['spike_data'][i]) > 10:
                    bins_size = float(np.mean(network_isi_distribution(
                        sim_data['spike_data'][i])))
                    av_times, av_sizes = avalanches_from_zero_activity(
                        sim_data['spike_data'][i],
                        0.0,
                        sim_data['duration'],
                        bins_size,
                        True
                    )
                    av_durations = av_times[:, 1] - av_times[:, 0]
                    sizes[i] = np.concatenate((sizes[i], av_sizes))
                    durations[i] = np.concatenate((durations[i], av_durations))
                else:
                    print("No spikes for ", mu)

    return sizes, durations, mus
