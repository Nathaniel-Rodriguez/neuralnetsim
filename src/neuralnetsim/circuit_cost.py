__all__ = ["circuit_cost"]


import numpy as np
from neuralnetsim import ArrayTranslator
from neuralnetsim import TrialManager
from neuralnetsim import CircuitParameters
from neuralnetsim import coincidence_factor
from typing import Dict


def circuit_cost(x: np.ndarray,
                 circuit_parameters: CircuitParameters,
                 translator: ArrayTranslator,
                 kernel_parameters: Dict,
                 data: Dict[int, np.ndarray],
                 run_duration: float,
                 coincidence_window: float) -> float:
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
    :return: The sum of the coincidence factors for all neurons in the model.
    """
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
    return sum(cost for cost in costs if not np.isnan(cost))
