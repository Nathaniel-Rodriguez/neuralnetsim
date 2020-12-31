__all__ = ["CircuitParameters",
           "DistributionParameters"]


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from neuralnetsim import get_translator
from neuralnetsim import ValueTranslator
from neuralnetsim import DistributionTranslator
from neuralnetsim import ArrayTranslator
from typing import Dict
from typing import Any
from typing import List
from typing import Union
from typing import Sequence


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
                 homogeneous_neurons: bool = False,
                 translator: ArrayTranslator = None):
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
        self._translator = translator

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

    def required_array_size(self) -> int:
        """
        :return: Size of the optimizer array required in order to satisfy the
            parameter conversion to models.
        """
        if self._translator is not None:
            return self._translator.required_array_size()
        else:
            raise RuntimeError

    def from_optimizer(self, array: np.ndarray):
        if self._translator is not None:
            self._translator.from_optimizer(array)
            self.extend_global_parameters(self._translator.global_parameters)
            self.extend_neuron_parameters(self._translator.neuron_parameters)
            self.extend_synaptic_parameters(self._translator.synapse_parameters)
            self.extend_noise_parameters(self._translator.noise_parameters)
        else:
            raise RuntimeError


class DistributionParameters:
    """"""
    def __init__(
            self,
            graph: nx.DiGraph,
            neuron_model: str,
            translators: List[Union[ValueTranslator, DistributionTranslator]],
            global_keys: Sequence[str],
            noise_keys: Sequence[str],
            synapse_keys: Sequence[str],
            neuron_keys: Sequence[str],
            static_neuron_parameters: Dict[str, Any] = None,
            static_synaptic_parameters: Dict[str, Any] = None,
            static_noise_parameters: Dict[str, Any] = None,
            static_global_parameters: Dict[str, Any] = None,
    ):
        self.network = graph
        self.neuron_model = neuron_model
        self._translators = translators
        self._global_keys = list(global_keys)
        self._noise_keys = list(noise_keys)
        self._synapse_keys = list(synapse_keys)
        self._neuron_keys = list(neuron_keys)
        self._key_order = self._global_keys \
                          + self._noise_keys \
                          + self._synapse_keys \
                          + self._neuron_keys

        self._array_size = 0
        for key in self._key_order:
            translator = get_translator(self._translators, key)
            self._array_size += translator.num_parameters()

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
        self._opt_parameters = np.zeros(self._array_size)
        self._dist_args = {}
        self._key_map = {key: self.neuron_parameters for key in self._neuron_keys}
        self._key_map.update({key: self.global_parameters for key in self._global_keys})
        self._key_map.update({key: self.synaptic_parameters for key in self._synapse_keys})
        self._key_map.update({key: self.noise_parameters for key in self._noise_keys})

    def generate_neuron_parameters(self, num_neurons: int,
                                   rng: np.random.RandomState) -> List[Dict[str, Any]]:
        """

        :param num_neurons:
        :return:
        """
        results = {}
        for key in self._neuron_keys:
            if key in self._dist_args:
                trans = get_translator(self._translators, key)
                calling = {kwarg: self._dist_args[key][i]
                           for i, kwarg in enumerate(trans.dist_args)}
                calling.update({'size': num_neurons})
                result = getattr(rng, trans.dist_type)(**calling) * trans.scale + trans.shift
                results[key] = result

        params = []
        for i in range(num_neurons):
            pars = self.neuron_parameters.copy()
            for key in self._neuron_keys:
                pars[key] = results[key][i]
            params.append(pars)
        return params

    def generate_synapse_parameters(self, num_synapses: int,
                                    rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        """

        :param num_synapses:
        :return: 1xN array for each distribution key, else
        """
        results = {}
        for key in self._synapse_keys:
            if key in self._dist_args:
                trans = get_translator(self._translators, key)
                calling = {kwarg: self._dist_args[key][i]
                           for i, kwarg in enumerate(trans.dist_args)}
                calling.update(size=(1, num_synapses))
                result = getattr(rng, trans.dist_type)(**calling) * trans.scale + trans.shift
                results[key] = result

        pars = self.synaptic_parameters.copy()
        pars.update(results)
        return pars

    def required_array_size(self) -> int:
        """
        :return: Size of the optimizer array required in order to satisfy the
            parameter conversion to models.
        """
        return self._array_size

    def from_optimizer(self, array: np.ndarray):
        if len(array) != self._array_size:
            raise ValueError("Optimizer array is wrong size. Expected {0},"
                             " received {1}.".format(str(self._array_size),
                                                     str(len(array))))
        self._opt_parameters = array
        index = 0
        for key in self._key_order:
            trans = get_translator(self._translators, key)
            if isinstance(trans, ValueTranslator):
                self._key_map[key][key] = trans.to_model(array[index])
            elif isinstance(trans, DistributionTranslator):
                self._dist_args[key] = trans.to_model(
                    *array[index:index+trans.num_parameters()])
            index += trans.num_parameters()

    def plot_parameter_distributions(
            self,
            outdir: Path = None,
            prefix: str = "test",
            num_samples: int = 10000,
            bins: int = 100
    ):
        if outdir is None:
            outdir = Path.cwd()
        rng = np.random.RandomState()
        for trans in self._translators:
            if isinstance(trans, DistributionTranslator):
                calling = {kwarg: self._dist_args[trans.key][i]
                           for i, kwarg in enumerate(trans.dist_args)}
                calling.update({'size': num_samples})
                result = getattr(rng, trans.dist_type)(**calling) * trans.scale + trans.shift
                plt.hist(result, bins=bins)
                plt.title(trans.key + " " + str(calling)
                          + " scale " + str(trans.scale) + " shift "
                          + str(trans.shift),
                          fontsize=6)
                plt.savefig(outdir.joinpath(prefix + "_" + trans.key + "_dist.png"),
                            dpi=300)
                plt.close()
                plt.clf()

    def __repr__(self):
        s = "======== Circuit Parameters ===============" + "\n"
        s += "network: " + self.network.__repr__() + "\n"
        s += nx.info(self.network) + "\n"
        s += "neuron model: " + self.neuron_model.__repr__() + "\n"
        s += "translators: " + self._translators.__repr__() + "\n"
        s += "global keys: " + self._global_keys.__repr__() + "\n"
        s += "noise keys: " + self._noise_keys.__repr__() + "\n"
        s += "synapse keys: " + self._synapse_keys.__repr__() + "\n"
        s += "neuron keys: " + self._neuron_keys.__repr__() + "\n"
        s += "key order: " + self._key_order.__repr__() + "\n"
        s += "array size: " + self._array_size.__repr__() + "\n"
        s += "dist args: " + self._dist_args.__repr__() + "\n"
        s += "key map: " + self._key_map.__repr__() + "\n"
        s += "opt parameters: " + self._opt_parameters.__repr__() + "\n"
        s += "============================================" + "\n"
        return s
