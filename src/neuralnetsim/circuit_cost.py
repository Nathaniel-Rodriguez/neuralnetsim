__all__ = ["circuit_cost",
           "training_cost",
           "avalanche_cost",
           "distributed_avalanche_cost",
           "avalanche_participation_cost",
           "ph_cost",
           "avalanche_firing_cost",
           "size_and_duration_cost",
           "duration_cost"]


import math
import numpy as np
import neuralnetsim
import umap
from neuralnetsim import ArrayTranslator
from neuralnetsim import TrialManager
from neuralnetsim import CircuitParameters
from neuralnetsim import coincidence_factor
from neuralnetsim import TrainingManager
from neuralnetsim import CircuitManager
from neuralnetsim import NeuralCircuit
from neuralnetsim import avalanches_from_median_activity
from neuralnetsim import DistributionParameters
from neuralnetsim import DistributionCircuit
from neuralnetsim import InhibCircuit
from scipy.stats import wasserstein_distance
from typing import Dict
from typing import List


def circuit_cost(x: np.ndarray,
                 circuit_parameters: CircuitParameters,
                 translator: ArrayTranslator,
                 kernel_parameters: Dict,
                 data: Dict[int, np.ndarray],
                 run_duration: float,
                 coincidence_window: float,
                 kernel_seeder: np.random.RandomState = None,
                 return_model_spikes: bool = False) -> float:
    """
    Calculates the cost for a given parameter array. Uses the coincidence
    factor to assess how well all model spikes matched against their respective
    real data counterparts.

    :param x: A 1-D array of the parameters.
    :param circuit_parameters: A parameter object that contains all static and
        trainable parameters. Trainable parameters will be set at runtime.
    :param translator: Will convert the parameter array into model parameters.
    :param kernel_parameters: Parameters for the NEST kernel.
    :param data: A dictionary keyed by node Id and valued by spike times.
    :param run_duration: How long to run the NEST simulation of the model (ms).
    :param coincidence_window: Size of the coincidence function window (ms).
    :param kernel_seeder: Optionally sets kernel seeds.
    :param return_model_spikes: Whether to return the model's spike times.
    :return: The sum of the coincidence factors for all neurons in the model.
    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    translator.from_optimizer(x)
    circuit_parameters.extend_global_parameters(translator.global_parameters)
    circuit_parameters.extend_neuron_parameters(translator.neuron_parameters)
    circuit_parameters.extend_synaptic_parameters(translator.synapse_parameters)
    circuit_parameters.extend_noise_parameters(translator.noise_parameters)
    with TrialManager(kernel_parameters, circuit_parameters, data) as circuit:
        circuit.run(run_duration)
        model_spikes = circuit.get_spike_trains()
        costs = [coincidence_factor([model_spikes[node]], data[node],
                                    run_duration, coincidence_window)
                 for node in model_spikes.keys()]

    cost = sum(cost for cost in costs if not np.isnan(cost))
    if return_model_spikes:
        return cost, model_spikes
    else:
        return cost


def training_cost(x: np.ndarray,
                  circuit_parameters: CircuitParameters,
                  translator: ArrayTranslator,
                  kernel_parameters: Dict,
                  training_manager: TrainingManager,
                  coincidence_window: float,
                  kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the coincidence
    factor to assess how well all model spikes matched against their respective
    real data counterparts. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer.

    :param x: A 1-D array of the parameters.
    :param circuit_parameters: A parameter object that contains all static and
        trainable parameters. Trainable parameters will be set at runtime.
    :param translator: Will convert the parameter array into model parameters.
    :param kernel_parameters: Parameters for the NEST kernel.
    :param training_manager: A TrainingManager.
    :param coincidence_window: Size of the coincidence function window (ms).
    :param kernel_seeder: Optionally sets kernel seeds.
    :return: The sum of the coincidence factors for all neurons in the model.
    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    translator.from_optimizer(x)
    circuit_parameters.extend_global_parameters(translator.global_parameters)
    circuit_parameters.extend_neuron_parameters(translator.neuron_parameters)
    circuit_parameters.extend_synaptic_parameters(translator.synapse_parameters)
    circuit_parameters.extend_noise_parameters(translator.noise_parameters)
    data = training_manager.get_training_data()
    with TrialManager(kernel_parameters, circuit_parameters, data) as circuit:
        try:
            circuit.run(training_manager.get_duration())
            model_spikes = circuit.get_spike_trains()
            costs = [coincidence_factor([model_spikes[node]], data[node],
                                        training_manager.get_duration(),
                                        coincidence_window)
                     for node in model_spikes.keys()]

        except Exception as err:  # nest just throws exceptions
            print(err)
            costs = [9e9]

    return sum(cost for cost in costs if not np.isnan(cost))


def avalanche_cost(
        x: np.ndarray,
        circuit_parameters: CircuitParameters,
        translator: ArrayTranslator,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    :param x: A 1-D array of the parameters.
    :param circuit_parameters: A parameter object that contains all static and
        trainable parameters. Trainable parameters will be set at runtime.
    :param translator: Will convert the parameter array into model parameters.
    :param kernel_parameters: Parameters for the NEST kernel.
    :param duration: Duration of the run.
    :param data_avalanche_sizes: The avalanche size distribution of the data.
    :param kernel_seeder: Optionally sets kernel seeds.
    :return: Wasserstein distance (cost) between model and data avalanche size
        distributions.
    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    translator.from_optimizer(x)
    circuit_parameters.extend_global_parameters(translator.global_parameters)
    circuit_parameters.extend_neuron_parameters(translator.neuron_parameters)
    circuit_parameters.extend_synaptic_parameters(translator.synapse_parameters)
    circuit_parameters.extend_noise_parameters(translator.noise_parameters)
    with CircuitManager(NeuralCircuit, kernel_parameters, circuit_parameters) as circuit:
        circuit.run(duration)
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 5000000):
            model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )[1]
            # _, data_avalanche_sizes = avalanches_from_median_activity(
            #     data,
            #     0.0,
            #     training_manager.get_duration()
            # )
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                cost = -1 / math.log(d + 1)  # max cost is 0, min cost -inf
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("Num of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def distributed_avalanche_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(DistributionCircuit, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    # 'max_spikes': 1250000000 / duration * 1000 # ~10GB
                    'max_spikes': 10000  # ~10 spikes/ms
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )[1]
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                cost = -1 / math.log(d + 1)  # max cost is 0, min cost -inf
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("Num of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def avalanche_participation_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        data_participation_dist: np.ndarray,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(DistributionCircuit, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spike': 10000
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds ~10GB return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            times, model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                cost = -1 / math.log(d + 1)  # max cost is 0, min cost -inf

                model_participation_dist = neuralnetsim.participating_neuron_distribution(
                    times, model_spikes
                )
                e = wasserstein_distance(model_participation_dist,
                                         data_participation_dist)
                ecost = -1 / math.log(d + 1)
                cost += ecost
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("Num of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def ph_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_diags: List[np.ndarray],
        dimensionality: int,
        max_lags: int,
        ripser_kwargs: Dict,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(DistributionCircuit, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 10000  # ~10 spikes/ms
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        print("BEFORE num_spikes", num_spikes, flush=True)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 5000000):
            binned_data, _ = neuralnetsim.bin_spikes(
                model_spikes,
                0.0,
                duration,
                float(np.mean(neuralnetsim.network_isi_distribution(
                    model_spikes
                )))
            )
            print("bin len", len(binned_data), flush=True)
            try:
                model_diags = neuralnetsim.generate_persistence_diagrams(
                    binned_data,
                    dimensionality,
                    max_lags,
                    **ripser_kwargs
                )
                print("diag sizes", len(model_diags[0]),
                      len(model_diags[1]), len(model_diags[2]), flush=True)
            except ValueError as err:
                print(err)
                model_diags = neuralnetsim.generate_persistence_diagrams(
                    binned_data,
                    dimensionality,
                    int(len(binned_data) / dimensionality),  # reduce delay to fit embedding
                    **ripser_kwargs
                )
            d = neuralnetsim.diagram_distances(model_diags, data_diags)
            cost = -1 / math.log(d + 1)  # max cost is 0, min cost -inf
        else:
            cost = 0.0
        print("Num of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def avalanche_firing_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        data_firing_rate_dist: np.ndarray,
        firing_rate_scale: float,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(DistributionCircuit, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    # 'max_spikes': 1250000000 / duration * 1000 # ~10GB
                    'max_spikes': 10000  # ~10 spikes/ms
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )[1]
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                f = wasserstein_distance(
                    neuralnetsim.firing_rate_distribution(
                        model_spikes, duration),
                    data_firing_rate_dist)
                cost = -1 / math.log(d + 1)\
                       - 1 / math.log(f * firing_rate_scale + 1)
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("Num of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def size_and_duration_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        data_avalanche_durations: np.ndarray,
        circuit_choice,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(circuit_choice, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    # 'max_spikes': 1250000000 / duration * 1000 # ~10GB
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            print("\tMemory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_times, model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                # d = wasserstein_distance(model_avalanche_sizes,
                #                          data_avalanche_sizes)
                # f = wasserstein_distance(model_times[:, 1] - model_times[:, 0],
                #                          data_avalanche_durations)
                # cost = -1 / math.log(f + d + 1)  # max cost is 0, min cost -inf
                # d = wasserstein_distance(model_avalanche_sizes,
                #                          data_avalanche_sizes)
                f = wasserstein_distance(model_times[:, 1] - model_times[:, 0],
                                         data_avalanche_durations)
                cost = -1 / math.log(f + 1)  # max cost is 0, min cost -inf
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("\tNum of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost


def duration_cost(
        x: np.ndarray,
        circuit_parameters: DistributionParameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_durations: np.ndarray,
        circuit_choice,
        rng: np.random.RandomState,
        kernel_seeder: np.random.RandomState = None) -> float:
    """
    Calculates the cost for a given parameter array. Uses the Wasserstein
    distance between model avalanche size distribution and data avalanche size
    distribution as a cost. Uses a training manager to use subsets of the
    training data for batches for each epoch of the trainer. Kernel seeder is
    used to set new kernel seeds for each execution of the cost function.

    """
    if kernel_seeder is not None:
        kernel_parameters.update({'grng_seed': kernel_seeder.randint(1, 2e5),
                                  'rng_seeds': [kernel_seeder.randint(1, 2e5)]})
    circuit_parameters.from_optimizer(x)
    with CircuitManager(circuit_choice, kernel_parameters, circuit_parameters,
                        rng) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            print("\tMemory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_times, model_avalanche_sizes = avalanches_from_median_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                f = wasserstein_distance(model_times[:, 1] - model_times[:, 0],
                                         data_avalanche_durations)
                cost = -1 / math.log(f + 1)  # max cost is 0, min cost -inf
            else:
                cost = 0.0
        else:
            cost = 0.0
        print("\tNum of spikes:",
              str(int(num_spikes / 1000)) + "K",
              "Cost", cost, flush=True)

    return cost
