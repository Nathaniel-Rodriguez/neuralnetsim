__all__ = ["circuit_cost",
           "training_cost"]


import numpy as np
from neuralnetsim import ArrayTranslator
from neuralnetsim import TrialManager
from neuralnetsim import CircuitParameters
from neuralnetsim import coincidence_factor
from neuralnetsim import TrainingManager
from typing import Dict


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
