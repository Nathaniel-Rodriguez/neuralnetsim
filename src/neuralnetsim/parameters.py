__all__ = ["CircuitParameters"]


import networkx as nx
from typing import Dict
from typing import Any
from typing import List
from typing import Union


class CircuitParameters:
    """
    CircuitParameters holds all parameters necessary to configure a neural
    circuit or optimizer network. Can be extended with optimization parameters
    from the ArrayTranslator.
    """
    def __init__(self, graph: nx.DiGraph,
                 neuron_model: str,
                 static_neuron_parameters: Dict[str, Any] = None,
                 static_synaptic_parameters: Dict[str, Any] = None,
                 static_noise_parameters: Dict[str, Any] = None,
                 static_global_parameters: Dict[str, Any] = None,
                 nodes: List[int] = None,
                 homogeneous_neurons: bool = False):
        """
        :param graph: Network associated with the parameterization.
        :param neuron_model: A NEST neuron model.
        :param static_neuron_parameters: Any neuron parameters that will be
            set across all neurons and not be subject to optimization.
        :param static_synaptic_parameters: Any synaptic parameters that will be
            set across all synapses and not be subject to optimization (ignore
            weights which are handled automatically by the neural circuits).
        :param static_noise_parameters: Any noise parameters that will be
            set across all noise generators and not be subject to optimization.
        :param static_global_parameters: Any global parameters that are not
            subject to optimization.
        :param nodes: A list of training node IDs from the graph (default: None).
            If set, this list will be used instead of all nodes for generating the
            parameters. Use when only a subset of the graphs neurons will be trained.
        :param homogeneous_neurons: Specify whether to share neuron parameters
            among neurons (default = False).
        """
        if nodes is not None:
            self._nodes = nodes
        else:
            self._nodes = list(graph.nodes())
        self.network = graph
        self.neuron_model = neuron_model
        self._homogeneous_neurons = homogeneous_neurons
        if not self._homogeneous_neurons:
            self.neuron_parameters = {
                neuron_id: {} if static_neuron_parameters is None
                else static_neuron_parameters
                for neuron_id in self._nodes}
        else:
            if static_neuron_parameters is None:
                self.neuron_parameters = {}
            else:
                self.neuron_parameters = static_neuron_parameters

        if static_synaptic_parameters is None:
            self.synaptic_parameters = {}
        else:
            self.synaptic_parameters = static_synaptic_parameters

        if static_noise_parameters is None:
            self.noise_parameters = {}
        else:
            self.noise_parameters = static_noise_parameters

        if static_global_parameters is None:
            self.global_parameters = {}
        else:
            self.global_parameters = static_global_parameters

    def extend_synaptic_parameters(self, parameters: Dict[str, Any]):
        self.synaptic_parameters.update(parameters)

    def extend_noise_parameters(self, parameters: Dict[str, Any]):
        self.noise_parameters.update(parameters)

    def extend_global_parameters(self, parameters: Dict[str, Any]):
        self.global_parameters.update(parameters)

    def extend_neuron_parameters(
            self,
            parameters: Union[Dict[int, Dict[str, Any]], Dict[str, Any]]):
        if not self._homogeneous_neurons:
            for node in self.neuron_parameters.keys():
                self.neuron_parameters[node].update(parameters[node])
        else:
            self.neuron_parameters.update(parameters)

    def training_nodes(self) -> List[int]:
        """
        :return: A list of node IDs for the neurons in the graph that will be
            trained.
        """
        return self._nodes

    def is_homogeneous(self) -> bool:
        """
        :return: True if neurons share parameters, else False.
        """
        return self._homogeneous_neurons
