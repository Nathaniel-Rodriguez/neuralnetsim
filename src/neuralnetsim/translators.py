__all__ = ["ValueTranslator", "ArrayTranslator",
           "get_translator"]


import math
import networkx as nx
import numpy as np
from typing import List
from typing import Dict
from typing import Any
from typing import Set


class ValueTranslator:
    """
    The ValueTranslator class will translate between model and optimizer
    parameters. A "key" property specifies variable type the translator is
    associated with. Bounds must be provided for the minimum and maximum allowed
    values for the model parameter. Model parameters are translated into a
    range of [0,1] and will automatically convert optimizer values into this
    range through periodic boundary conditions. Use the to_model and
    to_optimizer methods translate a given value for the model or for the
    optimizer.
    """
    def __init__(self, key: str, vmin: float, vmax: float):
        """
        :param key: A string specifying the kind of parameter this translator
        is responsible for.
        :param vmin: The minimum allowed model value for this parameter.
        :param vmax: The maximum allowed model value for this parameter.
        """
        self.key = key
        self._min = vmin
        self._max = vmax

    def to_model(self, optimizer_value: float) -> float:
        """
        Converts an optimizer value into a model value.
        :param optimizer_value: A given optimizer value.
        :return: The converted value for the model.
        """
        return ((-math.fabs(optimizer_value % 2 - 1) + 1)
                * (self._max - self._min)) + self._min

    def to_optimizer(self, model_value: float) -> float:
        """
        Converts a model value into an optimizer value.
        :param model_value: A given model value.
        :return: The converted value for the optimizer.
        """
        return (model_value - self._min) / (self._max - self._min)

    def __eq__(self, other):
        return self.key == other.key\
               and self._min == other._min\
               and self._max == other._max


def get_translator(translators: List[ValueTranslator],
                   key: str) -> ValueTranslator:
    """
    Finds and returns a translator with a given key from a list.
    :param translators: A list of translators to search.
    :param key: A translator key.
    :return: A translator, or raises a KeyError if none is found.
    """
    for translator in translators:
        if translator.key == key:
            return translator
    raise KeyError("Failed to find translator in list with key: " + key)


class ArrayTranslator:
    """
    ArrayTranslator will convert optimizer parameter arrays into the model
    parameters or vice versa.
    """
    def __init__(self, graph: nx.DiGraph, translators: List[ValueTranslator],
                 neuron_keys: Set[str], synapse_keys: Set[str],
                 noise_keys: Set[str], global_keys: Set[str]):
        """
        :param graph: The graph used for the neural circuit optimization.
        :param translators: A list of translators with keys associated with
        all parameters that will be optimized.
        :param neuron_keys: A set of keys associated with NEST neuron models.
        :param synapse_keys: A set of keys associated with NEST synapse models.
        :param noise_keys: A set of keys associated with NEST noise models.
        :param global_keys: A set of global keys.
        """
        self._graph = graph
        self._translators = translators
        self._neuron_keys = neuron_keys
        self._synapse_keys = synapse_keys
        self._noise_keys = noise_keys
        self._global_keys = global_keys

        self._neuron_parameters = {neuron_id: {} for neuron_id in graph.nodes()}
        self._synapse_parameters = {}
        self._noise_parameters = {}
        self._global_parameters = {}

        self._key_order = list(self._global_keys)\
                          + list(self._noise_keys)\
                          + list(self._synapse_keys)\
                          + [key
                             for _ in range(nx.number_of_nodes(graph))
                             for key in self._neuron_keys]
        self._array_size = len(self._key_order)
        self._model_parameters = np.zeros(self._array_size)

    @property
    def synapse_parameters(self) -> Dict[str, Any]:
        """
        :return: A dictionary of the synaptic parameters for the NEST model.
        """
        return self._synapse_parameters

    @property
    def neuron_parameters(self) -> Dict[int, Dict[str, Any]]:
        """
        :return: A dictionary keyed by neuron id and valued by model parameters
        for neuron models.
        """
        return self._neuron_parameters

    @property
    def noise_parameters(self) -> Dict[str, Any]:
        """
        :return: A dictionary keyed by NEST model parameter name and valued by
        the corresponding value.
        """
        return self._noise_parameters

    @property
    def global_parameters(self) -> Dict[str, Any]:
        """
        :return: A dictionary keyed by global parameter names.
        """
        return self._global_parameters

    def required_array_size(self) -> int:
        """
        :return: Size of the optimizer array required in order to satisfy the
        parameter conversion to models.
        """
        return self._array_size

    def from_optimizer(self, array: np.ndarray):
        """
        Set model parameters in-place when given an optimization array.
        :param array: An array whose elements can be consumed by ValueTranslators.
        :return: None
        """
        if len(array) != self._array_size:
            raise ValueError("Optimizer array is wrong size. Expected {0},"
                             " received {1}.".format(str(self._array_size),
                                                     str(len(array))))
        for i, opt_value in enumerate(array):
            self._model_parameters[i] = get_translator(
                self._translators, self._key_order[i]).to_model(opt_value)

        c = 0
        for key in self._global_keys:
            self._global_parameters[key] = self._model_parameters[c]
            c += 1
        for key in self._noise_keys:
            self._noise_parameters[key] = self._model_parameters[c]
            c += 1
        for key in self._synapse_keys:
            self._synapse_parameters[key] = self._model_parameters[c]
            c += 1
        for neuron_pars in self._neuron_parameters.values():
            for key in self._neuron_keys:
                neuron_pars[key] = self._model_parameters[c]
                c += 1

    def to_optimizer(self) -> np.ndarray:
        """
        :return: An array consumable by an optimizer. Should not be run unless
        from_optimizer has initialized the internal model parameters.
        """
        opt_array = np.zeros(self._array_size)
        for i, key in enumerate(self._key_order):
            opt_array[i] = get_translator(
                self._translators, key).to_optimizer(self._model_parameters[i])
        return opt_array
