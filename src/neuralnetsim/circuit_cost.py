__all__ = ["circuit_cost",
           "training_cost",
           "avalanche_cost",
           "distributed_avalanche_cost",
           "avalanche_participation_cost",
           "ph_cost",
           "avalanche_firing_cost",
           "size_and_duration_cost",
           "duration_cost",
           "map_cost",
           "map_cost2",
           "map_cost3",
           "map_cost4",
           "IzhikevichMemberSource"]


import math
import numpy as np
import neuralnetsim
from neuralnetsim import ArrayTranslator
from neuralnetsim import TrialManager
from neuralnetsim import CircuitParameters
from neuralnetsim import adjusted_coincidence_factor
from neuralnetsim import TrainingManager
from neuralnetsim import CircuitManager
from neuralnetsim import NeuralCircuit
from neuralnetsim import avalanches_from_zero_activity
from neuralnetsim import DistributionParameters
from neuralnetsim import DistributionCircuit
from scipy.stats import wasserstein_distance
from typing import Dict
from typing import Tuple
from typing import List


class IzhikevichMemberSource(neuralnetsim.DrawMember):
    def __init__(
            self,
            parameters: neuralnetsim.AllNeuronSynDistParameters,
            inhib_fract=0.2,
            disable_inhib=False
    ):
        self._ndim = parameters.required_array_size()
        self._inhib_fract = inhib_fract
        self._disable_inhib = disable_inhib
        self._num_neurons = len(parameters.network)
        self._a_indices = []
        self._b_indices = []
        self._c_indices = []
        self._d_indices = []
        self._mode_indices = []
        index = 0
        for key in parameters.key_order:
            if key == "a":
                self._a_indices.append(index)
            elif key == "b":
                self._b_indices.append(index)
            elif key == "c":
                self._c_indices.append(index)
            elif key == "d":
                self._d_indices.append(index)
            elif key == "mode":
                self._mode_indices.append(index)
            index += neuralnetsim.get_translator(
                parameters.translators, key).num_parameters()
        # from NEST defaults
        self._default_a = neuralnetsim.get_translator(parameters.translators, "a").to_optimizer(0.02)
        self._default_b = neuralnetsim.get_translator(parameters.translators, "b").to_optimizer(0.2)
        self._default_c = neuralnetsim.get_translator(parameters.translators, "c").to_optimizer(-65.0)
        self._default_d = neuralnetsim.get_translator(parameters.translators, "d").to_optimizer(8.0)

        self._default_in = 0.49#0.25
        self._default_ex = 0.51#0.75

    def draw(self, rng: np.random.RandomState) -> np.ndarray:
        random_member = rng.uniform(0.0, 1.0, size=self._ndim)
        # set appropriate indices with defaults
        random_member[self._a_indices] = self._default_a
        random_member[self._b_indices] = self._default_b
        random_member[self._c_indices] = self._default_c
        random_member[self._d_indices] = self._default_d

        if not self._disable_inhib:
            in_indices = rng.choice(self._mode_indices,
                                    int(self._inhib_fract * len(self._mode_indices)),
                                    replace=False)
            ex_indices = [index for index in self._mode_indices
                          if index not in in_indices]
            random_member[in_indices] = self._default_in
            random_member[ex_indices] = self._default_ex
        return random_member


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
        costs = [adjusted_coincidence_factor([model_spikes[node]], data[node],
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
            costs = [adjusted_coincidence_factor([model_spikes[node]], data[node],
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
            model_avalanche_sizes = avalanches_from_zero_activity(
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
            model_avalanche_sizes = avalanches_from_zero_activity(
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
        data_participation_dist: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> float:
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
                        **kwargs) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spike': 16000
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds ~10GB return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                model_participation_dist = neuralnetsim.participating_neuron_distribution(
                    times, model_spikes
                )
                p = wasserstein_distance(model_participation_dist,
                                         data_participation_dist)
                cost = -1 / math.log(p + 1)
            else:
                cost = 0.0
        else:
            cost = 0.0

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
            # print("Memory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_avalanche_sizes = avalanches_from_zero_activity(
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
        data_firing_rate_dist: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> float:
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
                        **kwargs) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            # print("\tMemory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                f = wasserstein_distance(model_times[:, 1] - model_times[:, 0],
                                         data_avalanche_durations)
                r = wasserstein_distance(
                    neuralnetsim.firing_rate_distribution(
                        model_spikes, duration
                    ),
                    data_firing_rate_dist
                )
                cost = -1 / math.log(1000.0 * r + f + d + 1)  # max cost is 0, min cost -inf
                print("\t\tdistances:", 1000.0*r, f, d)
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
        circuit_parameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_durations: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> float:
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
                        **kwargs) as circuit:
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            # print("\tMemory guard activated", flush=True)
            return 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
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


def map_cost(
        x: np.ndarray,
        circuit_parameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> Tuple[float, float, float]:
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
                        **kwargs) as circuit:
        f1 = 0.0
        f2 = 0.0
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            # print("\tMemory guard activated", flush=True)
            return 0.0, 1.1, 1e6  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            f1 = neuralnetsim.percent_active(model_spikes)
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                f = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                cost = -1 / math.log(f + 1)  # max cost is 0, min cost -inf
                f2 = np.max(model_times[:, 1] - model_times[:, 0])
            else:
                cost = 0.0
        else:
            cost = 0.0
        # print("\tNum of spikes:",
        #       str(int(num_spikes / 1000)) + "K",
        #       "Cost", cost, flush=True)

    return cost, f1, f2


def map_cost2(
        x: np.ndarray,
        circuit_parameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        data_avalanche_durations: np.ndarray,
        data_firing_rate_dist: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> Tuple[float, float, float]:
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
                        **kwargs) as circuit:
        f1 = 0.0
        f2 = 0.0
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            # print("\tMemory guard activated", flush=True)
            return 0.0, 0.0, 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            f1 = neuralnetsim.percent_active(model_spikes)
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                f = wasserstein_distance(model_times[:, 1] - model_times[:, 0],
                                         data_avalanche_durations)
                r = wasserstein_distance(
                    neuralnetsim.firing_rate_distribution(
                        model_spikes, duration
                    ),
                    data_firing_rate_dist
                )
                cost = -1 / math.log(1000.0 * r + f + d + 1)  # max cost is 0, min cost -inf
                f2 = np.log10(np.max(model_times[:, 1] - model_times[:, 0]))
            else:
                cost = 0.0
        else:
            cost = 0.0
        # print("\tNum of spikes:",
        #       str(int(num_spikes / 1000)) + "K",
        #       "Cost", cost, flush=True)

    return cost, f1, f2


def map_cost3(
        x: np.ndarray,
        circuit_parameters,
        kernel_parameters: Dict,
        duration: float,
        data_avalanche_sizes: np.ndarray,
        data_avalanche_durations: np.ndarray,
        data_firing_rate_dist: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> Tuple[float, float, float]:
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
                        **kwargs) as circuit:
        f1 = 0.0
        f2 = 0.0
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 8000  # ~10 spikes/ms
                }
        ):
            print("\tMemory guard activated", flush=True)
            return 0.0, 0.0, 0.0  # if expected number of spikes exceeds limit return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            f1 = neuralnetsim.percent_active(model_spikes)
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                model_durations = model_times[:, 1] - model_times[:, 0]
                d = wasserstein_distance(model_avalanche_sizes,
                                         data_avalanche_sizes)
                f = wasserstein_distance(model_durations,
                                         data_avalanche_durations)
                r = wasserstein_distance(
                    neuralnetsim.firing_rate_distribution(
                        model_spikes, duration
                    ),
                    data_firing_rate_dist
                )
                cost = -1 / math.log(1000.0 * r + f + d + 1)  # max cost is 0, min cost -inf

                print("\t\tf1: ", f1, "f2: ", f2, "cost:", cost, flush=True)
            else:
                cost = 0.0
        else:
            cost = 0.0
        # print("\tNum of spikes:",
        #       str(int(num_spikes / 1000)) + "K",
        #       "Cost", cost, flush=True)

    return cost, f1, f2


def map_cost4(
        x: np.ndarray,
        circuit_parameters,
        kernel_parameters: Dict,
        duration: float,
        data_participation_dist: np.ndarray,
        data_durations: np.ndarray,
        circuit_choice,
        kernel_seeder: np.random.RandomState = None,
        **kwargs
) -> Tuple[float, float, float]:
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
                        **kwargs) as circuit:
        f1 = 0.0
        f2 = 0.0
        if not circuit.run(
                duration,
                memory_guard={
                    'duration': 1000.0,
                    'max_spikes': 16000  # ~10 spikes/ms
                }
        ):
            print("Memory guard activated", flush=True)
            return 0.0, 0.0, 0.0  # if expected number of spikes exceeds ~10GB return
        model_spikes = circuit.get_spike_trains()
        num_spikes = neuralnetsim.spike_count(model_spikes)
        # if too few spikes, then isi can't be calculated, if too many
        # than too much memory is used.
        if (num_spikes > 4) and (num_spikes < 10000000):
            f1 = neuralnetsim.percent_active(model_spikes)
            model_times, model_avalanche_sizes = avalanches_from_zero_activity(
                model_spikes,
                0.0,
                duration
            )
            if len(model_avalanche_sizes) > 0:
                model_participation_dist = neuralnetsim.participating_neuron_distribution(
                    model_times, model_spikes
                )
                p = wasserstein_distance(model_participation_dist,
                                         data_participation_dist)
                model_durations = model_times[:, 1] - model_times[:, 0]
                f = wasserstein_distance(model_durations,
                                         data_durations)
                cost = -1 / math.log(f / 3 + p + 1)
                # cost = -1 / math.log(p + 1)
                # print("Cost", cost, "d", f, "p", p, flush=True)
                f2 = np.log10(np.max(model_times[:, 1] - model_times[:, 0]))
            else:
                cost = 0.0
        else:
            cost = 0.0

    return cost, f1, f2
