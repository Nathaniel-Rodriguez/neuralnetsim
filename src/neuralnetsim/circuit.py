__all__ = ["NeuralCircuit",
           "DistributionCircuit",
           "CircuitManager",
           "InhibCircuit"]


import copy
import networkx as nx
import numpy as np
import nest
nest.set_verbosity("M_FATAL")
from typing import Dict
from typing import Any
from typing import Type
from typing import Union
from neuralnetsim import CircuitParameters
from neuralnetsim import DistributionParameters
from neuralnetsim import AllNeuronSynDistParameters
from neuralnetsim import spike_count


class NeuralCircuit:
    """
    Creates a neural circuit based on circuit parameters. Does not take inputs.
    """
    def __init__(self, circuit_parameters: CircuitParameters):
        """
        :param circuit_parameters: The parameters for the neural circuit.
        """
        self._circuit_parameters = circuit_parameters
        graph = self._circuit_parameters.network

        # create neurons
        if not self._circuit_parameters.is_homogeneous():
            self._neurons = nest.Create(
                self._circuit_parameters.neuron_model,
                n=nx.number_of_nodes(graph),
                params=list(self._circuit_parameters.neuron_parameters.values()))
        else:
            self._neurons = nest.Create(
                self._circuit_parameters.neuron_model,
                n=nx.number_of_nodes(graph),
                params=self._circuit_parameters.neuron_parameters)
        self._neuron_to_nest = {neuron_id: self._neurons[i]
                                for i, neuron_id in enumerate(graph.nodes())}
        self._nest_to_neuron = {self._neurons[i]: neuron_id
                                for i, neuron_id in enumerate(graph.nodes())}

        # create noise
        self._noise = nest.Create(
            "poisson_generator",
            n=1,
            params=self._circuit_parameters.noise_parameters)

        # create detectors
        self._detectors = nest.Create("spike_detector",
                                      n=nx.number_of_nodes(graph))
        self._detector_map = {neuron_id: self._detectors[i]
                              for i, neuron_id in enumerate(graph.nodes())}

        # connect nodes
        nest.Connect(self._noise, self._neurons, 'all_to_all')
        nest.Connect(self._neurons, self._detectors, 'one_to_one')
        for neuron_id in graph.nodes():
            presynaptic_neurons = list(graph.predecessors(neuron_id))
            presynaptic_weights = np.array([[
                weight
                for edge, weight in nx.get_edge_attributes(graph, "weight").items()
                if (edge[0] in presynaptic_neurons)
                   and (edge[1] == neuron_id)]])
            synaptic_params = self._circuit_parameters.synaptic_parameters.copy()
            synaptic_params['weight'] = \
                presynaptic_weights \
                * self._circuit_parameters.global_parameters['weight_scale']
            if len(presynaptic_neurons) > 0:
                nest.Connect([self._neuron_to_nest[neuron]
                              for neuron in presynaptic_neurons],
                             [self._neuron_to_nest[neuron_id]],
                             'all_to_all',
                             synaptic_params)

    def run(self, duration: float, memory_guard: Dict = None) -> bool:
        """
        Prompts the NEST kernel to run the simulation.

        :param duration: How long to run the simulation in ms.
        :param memory_guard: Does an initial simulation run to check if too many
            spikes are going to be produced. If a threshold is exceeded then
            the full simulation is cancelled and False is returned. Default is
            None. Expects a dictionary with keys 'duration' and 'max_spikes'.
            The first specifies how long the sample run lasts and the second is
            the spike cap.
        """
        if memory_guard is None:
            nest.Simulate(duration)
            return True
        else:
            with nest.RunManager():
                nest.Run(memory_guard['duration'])
                if spike_count(self.get_spike_trains()) > memory_guard['max_spikes']:
                    return False
                else:
                    nest.Run(duration - memory_guard['duration'])
                    return True

    def get_spike_trains(self) -> Dict[int, np.ndarray]:
        """
        :return: Spike train results from the simulation keyed by neuron id and
            valued by a 1-D numpy array of spike times.
        """
        return {
            neuron_id: nest.GetStatus([detector_id])[0]['events']['times']
            for neuron_id, detector_id in self._detector_map.items()
        }


class DistributionCircuit:
    """
    Creates a neural circuit based on circuit parameters. Does not take inputs.
    """
    def __init__(self, circuit_parameters: DistributionParameters,
                 rng: np.random.RandomState):
        """
        :param circuit_parameters: The parameters for the neural circuit.
        """
        self._rng = rng
        self._circuit_parameters = circuit_parameters
        graph = self._circuit_parameters.network

        # create neurons
        if isinstance(self._circuit_parameters, DistributionParameters):
            self._neurons = nest.Create(
                self._circuit_parameters.neuron_model,
                n=nx.number_of_nodes(graph),
                params=self._circuit_parameters.generate_neuron_parameters(
                    nx.number_of_nodes(graph),
                    self._rng
                ))
        elif isinstance(self._circuit_parameters, AllNeuronSynDistParameters):
            self._neurons = nest.Create(
                self._circuit_parameters.neuron_model,
                n=nx.number_of_nodes(graph),
                params=list(self._circuit_parameters.neuron_parameters.values()))
        self._neuron_to_nest = {neuron_id: self._neurons[i]
                                for i, neuron_id in enumerate(graph.nodes())}
        self._nest_to_neuron = {self._neurons[i]: neuron_id
                                for i, neuron_id in enumerate(graph.nodes())}

        # create noise
        self._noise = nest.Create(
            "poisson_generator",
            n=1,
            params=self._circuit_parameters.noise_parameters)

        # create detectors
        self._detectors = nest.Create("spike_detector",
                                      n=nx.number_of_nodes(graph))
        self._detector_map = {neuron_id: self._detectors[i]
                              for i, neuron_id in enumerate(graph.nodes())}

        # connect nodes
        nest.Connect(self._noise, self._neurons, 'all_to_all')
        nest.Connect(self._neurons, self._detectors, 'one_to_one')
        for neuron_id in graph.nodes():
            presynaptic_neurons = list(graph.predecessors(neuron_id))
            presynaptic_weights = np.array([[
                weight
                for edge, weight in nx.get_edge_attributes(graph, "weight").items()
                if (edge[0] in presynaptic_neurons)
                   and (edge[1] == neuron_id)]])
            synaptic_params = self._circuit_parameters.generate_synapse_parameters(
                len(presynaptic_neurons),
                self._rng
            )
            synaptic_params['weight'] = \
                presynaptic_weights \
                * self._circuit_parameters.global_parameters['weight_scale']
            res = nest.GetKernelStatus()['resolution']
            if 'delay' in synaptic_params:
                for i in range(len(synaptic_params['delay'][0])):
                    if synaptic_params['delay'][0][i] < res:
                        synaptic_params['delay'][0][i] = res
            if len(presynaptic_neurons) > 0:
                nest.Connect([self._neuron_to_nest[neuron]
                              for neuron in presynaptic_neurons],
                             [self._neuron_to_nest[neuron_id]],
                             'all_to_all',
                             synaptic_params)

    def run(self, duration: float, memory_guard: Dict = None) -> bool:
        """
        Prompts the NEST kernel to run the simulation.

        :param duration: How long to run the simulation in ms.
        :param memory_guard: Does an initial simulation run to check if too many
            spikes are going to be produced. If a threshold is exceeded then
            the full simulation is cancelled and False is returned. Default is
            None. Expects a dictionary with keys 'duration' and 'max_spikes'.
            The first specifies how long the sample run lasts and the second is
            the spike cap.
        """
        if memory_guard is None:
            nest.Simulate(duration)
            return True
        else:
            with nest.RunManager():
                nest.Run(memory_guard['duration'])
                if spike_count(self.get_spike_trains()) > memory_guard['max_spikes']:
                    return False
                else:
                    nest.Run(duration - memory_guard['duration'])
                    return True

    def get_spike_trains(self) -> Dict[int, np.ndarray]:
        """
        :return: Spike train results from the simulation keyed by neuron id and
            valued by a 1-D numpy array of spike times.
        """
        return {
            neuron_id: nest.GetStatus([detector_id])[0]['events']['times']
            for neuron_id, detector_id in self._detector_map.items()
        }


class InhibCircuit(DistributionCircuit):
    def __init__(
            self,
            circuit_parameters: DistributionParameters,
            rng: np.random.RandomState
    ):
        # parameters = copy.deepcopy(circuit_parameters)
        # # copy circuit parameters, split neuron parameters into ex/in
        # # make network weights inhibitory before initializing the circuit
        # for node, pars in parameters.neuron_parameters.items():
        #     if pars['mode'] < 0.5:  # if less than 0.5 treat as inhibitory
        #         successors = parameters.network.neighbors(node)
        #         for successor in successors:
        #             parameters.network[node][successor]['weight'] =\
        #                 -parameters.network[node][successor]['weight']
        #     del parameters.neuron_parameters[node]['mode']
        # super().__init__(parameters, rng)
        #
        parameters = copy.deepcopy(circuit_parameters)
        # copy circuit parameters, split neuron parameters into ex/in
        # make network weights inhibitory before initializing the circuit
        for node, pars in parameters.neuron_parameters.items():
            if rng.random() < 0.15:
                successors = parameters.network.neighbors(node)
                for successor in successors:
                    parameters.network[node][successor]['weight'] =\
                        -parameters.network[node][successor]['weight']
        super().__init__(parameters, rng)


class CircuitManager:
    """
    CircuitManager manages the NEST kernel as a context manager. It builds
    and returns an NeuralCircuit for the kernel and then cleans up the
    kernel after simulation is complete. Can be used to set the kernel seed
    or any other kernel parameters for the simulation. NeuralCircuit
    can only meaningfully exist within some kernel context.
    """
    def __init__(self,
                 circuit_type: Union[Type[DistributionCircuit], Type[NeuralCircuit]],
                 kernel_parameters: Dict[str, Any] = None,
                 *args, **kwargs):
        """
        :param kernel_parameters: Parameters forwarded to the NEST kernel.
        :param args: Arguments forwarded to the NeuralCircuit.
        :param kwargs: Keyword arguments forwarded to the NeuralCircuit.
        """
        self._circuit_type = circuit_type
        self._kernel_parameters = kernel_parameters
        self._circuit_args = args
        self._circuit_kwargs = kwargs

    def __enter__(self):
        nest.ResetKernel()  # Kernel must be reset before new simulation
        if self._kernel_parameters is not None:
            nest.SetKernelStatus(self._kernel_parameters)
        self.net = self._circuit_type(*self._circuit_args, **self._circuit_kwargs)
        return self.net

    def __exit__(self, exc_type, exc_val, exc_tb):
        nest.ResetKernel()


if __name__ == "__main__":
    pass

