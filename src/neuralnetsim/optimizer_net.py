__all__ = ["SubNetwork",
           "OptimizerNetwork"]


import networkx as nx
import numpy as np
import nest
from typing import Dict
from typing import List


class SubNetwork:
    """
    Given a directed graph and a target node, a subnetwork will be generated
    that allows for the simulation of that target node's spiking activity
    based on presynaptic activity. The presynaptic connections will be
    derived from the graph. Inputs must be provided independently and be in
    the order of the presynaptic nodes.
    """
    def __init__(self,
                 graph: nx.DiGraph,
                 node_id: int,
                 neuron_parameters: Dict,
                 synaptic_parameters: Dict,
                 noise_parameters: Dict,
                 neuron_model: str,
                 weight_scale: float,
                 inputs: List[np.ndarray] = None):
        """
        Initializes the subnetwork with the given parameters. It creates a
        network of a single model neuron with parrot neurons for all its
        presynaptic neighbors. Signals can be fed into these neighbors and
        will be propagated down to the model neuron. Which neuron to use must
        be specified by id from the provided graph.
        :param graph: The full directed graph. Must have a "weight" key for
        the edge attributes.
        :param node_id: The target neuron to build the subnetwork around.
        :param neuron_parameters: A dictionary of parameters keyed by model
        nest attributes and valued accordingly.
        :param synaptic_parameters: A dictionary of parameters keyed by
        synapse model attributes and valued accordingly. If a non-default
        model is desired, the "model" key must be specified in the dictionary.
        :param noise_parameters: A dictionary of parameters keyed for the
        nest noise_generator.
        :param neuron_model: Specifies the nest neuron model.
        :param weight_scale: A scalar factor for adjusting the weights.
        :param inputs: Optionally set the inputs into the neural network. If set
        should be a list of spike times for each presynaptic input into the
        target neuron.
        """
        self._graph_node_id = node_id
        self._presynaptic_nodes = list(graph.predecessors(self._graph_node_id))
        self._presynaptic_weights = np.array([[
            weight
            for edge, weight in nx.get_edge_attributes(graph, "weight").items()
            if (edge[0] in self._presynaptic_nodes)
               and (edge[1] == self._graph_node_id)]])

        # create nodes
        self._neuron = nest.Create(neuron_model, n=1,
                                   params=neuron_parameters)
        self._parrots = nest.Create("parrot_neuron",
                                    n=len(self._presynaptic_nodes))
        self._noise = nest.Create("noise_generator", n=1,
                                  params=noise_parameters)
        self._detector = nest.Create("spike_detector")
        self._spike_generators = nest.Create(
            "spike_generator", len(self._presynaptic_nodes),
            {'allow_offgrid_times': True})

        # make connections
        self._presynaptic_connections = nest.Connect(
            self._parrots, self._neuron, "all_to_all",
            self._append_weights(synaptic_parameters, weight_scale))
        self._noise_connection = nest.Connect(self._noise,
                                              self._neuron)
        self._detector_connection = nest.Connect(self._neuron,
                                                 self._detector)
        self._signal_connections = nest.Connect(
            self._spike_generators,
            self._parrots,
            "one_to_one",
            {"model": "static_synapse", "weight": 1.0, "delay": 0.1})

        if inputs is not None:
            self.set_inputs(inputs)

    def _append_weights(self, synaptic_parameters: Dict, weight_scale: float):
        """
        Adds weights for the corresponding
        :param synaptic_parameters: A dictionary of synapse parameters.
        :param weight_scale: A scalar to adjust the weight's scale.
        :return: The dictionary with a "weight" key added.
        """
        synaptic_parameters["weight"] = self._presynaptic_weights * weight_scale
        return synaptic_parameters

    def set_inputs(self, input_list: List[np.ndarray]):
        """
        Updates the spike generators with the new spike times. List should
        be the same length as the number of presynaptic neurons and should
        match the presynaptic neuron order.
        Spike times should be adjusted such that the first spike occurs at
        the time origin (0.1 ms).
        :param input_list: A list of numpy arrays of spike times.
        :return: None
        """
        if len(input_list) != len(self._presynaptic_nodes):
            raise AssertionError("Number of inputs into subnetwork did not"
                                 " match the number of presynaptic connections")
        nest.SetStatus(self._spike_generators,
                       [{'spike_times': inputs} for inputs in input_list])

    def update_network_parameters(self, neuron_parameters: Dict,
                                  synapse_parameters: Dict,
                                  noise_parameters: Dict,
                                  weight_scale: float):
        """
        Updates the target neuron, the presynaptic connections, and the noise
        generator.
        :param neuron_parameters: A dictionary with new settings for the
        neuron model.
        :param synapse_parameters: A dictionary with new settings for the
        synapse model.
        :param noise_parameters: A dictionary with new settings for the noise
        model.
        :param weight_scale: A scale factor for the weights.
        :return: None
        """
        nest.SetStatus(self._neuron, neuron_parameters)
        nest.SetStatus(self._presynaptic_connections,
                       self._append_weights(synapse_parameters, weight_scale))
        nest.SetStatus(self._noise, noise_parameters)

    def get_spike_output(self) -> np.ndarray:
        """
        :return: A numpy array of spike times for the target neuron.
        """
        return nest.GetStatus(self._detector)[0]['events']['times']

    @property
    def presynaptic_nodes(self) -> List[int]:
        return self._presynaptic_nodes

    @presynaptic_nodes.setter
    def presynaptic_nodes(self, v):
        raise NotImplementedError

    def __str__(self):
        s = "="*7 + "NEURON" + "="*7 + "\n"
        s += str(nest.GetStatus(self._neuron)) + "\n"
        s += "="*7 + "DETECTOR" + "="*7 + "\n"
        s += str(nest.GetStatus(self._detector)) + "\n"
        s += "="*7 + "NOISE" + "="*7 + "\n"
        s += str(nest.GetStatus(self._noise)) + "\n"
        s += "="*7 + "GENERATORS" + "="*7 + "\n"
        s += "\n".join(map(str, (out for out in nest.GetStatus(self._spike_generators))))
        return s


class OptimizerNetwork:
    def __init__(self):
        pass
