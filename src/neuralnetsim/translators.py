__all__ = ["ValueTranslator",
           "ArrayTranslator",
           "get_translator",
           "DistributionTranslator"]


import math
import networkx as nx
import numpy as np
from typing import List
from typing import Dict
from typing import Any
from typing import Sequence
from typing import Union
from typing import Tuple


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
    def __init__(self, key: str, vmin: float, vmax: float, log_scale=False):
        """
        :param key: A string specifying the kind of parameter this translator
            is responsible for.
        :param vmin: The minimum allowed model value for this parameter.
        :param vmax: The maximum allowed model value for this parameter.
        """
        self.key = key
        self._logged = log_scale
        if self._logged:
            self._min = math.log(vmin)
            self._max = math.log(vmax)
        else:
            self._min = vmin
            self._max = vmax

    def to_model(self, optimizer_value: float) -> float:
        """
        Converts an optimizer value into a model value.

        :param optimizer_value: A given optimizer value.
        :return: The converted value for the model.
        """
        value = ((-math.fabs(optimizer_value % 2 - 1) + 1)
                 * (self._max - self._min)) + self._min
        if self._logged:
            return math.exp(value)
        else:
            return value

    def to_optimizer(self, model_value: float) -> float:
        """
        Converts a model value into an optimizer value.

        :param model_value: A given model value.
        :return: The converted value for the optimizer.
        """
        if self._logged:
            return (math.log(model_value) - self._min) / (self._max - self._min)
        else:
            return (model_value - self._min) / (self._max - self._min)

    def num_parameters(self) -> int:
        return 1

    def __eq__(self, other):
        return self.key == other.key\
               and self._min == other._min\
               and self._max == other._max


class DistributionTranslator:
    def __init__(self, key: str, dist_type: str,
                 dist_bounds: Dict[str, Tuple[float, float]],
                 shift: float = None,
                 scale: float = None):
        """
        :param key: A string specifying the kind of parameter this translator
            is responsible for.
        :param vmin: The minimum allowed model value for this parameter.
        :param vmax: The maximum allowed model value for this parameter.
        """
        self.key = key
        self.dist_type = dist_type
        self.dist_args = [key for key in dist_bounds.keys()]
        self.shift = shift
        self.scale = scale
        self._dist_bounds = dist_bounds

    def to_model(self, *args) -> List[float]:
        """
        Converts an optimizer value into a model value.

        :param optimizer_value: A given optimizer value.
        :return: The converted value for the model.
        """
        dist_args = []
        for i, arg in enumerate(args):
            dist_args.append(((-math.fabs(arg % 2 - 1) + 1)
                     * (self._dist_bounds[self.dist_args[i]][1]
                        - self._dist_bounds[self.dist_args[i]][0]))
                             + self._dist_bounds[self.dist_args[i]][0])
        return dist_args

    # def to_optimizer(self, model_value: float) -> float:
    #     """
    #     Converts a model value into an optimizer value.
    #
    #     :param model_value: A given model value.
    #     :return: The converted value for the optimizer.
    #     """
    #     return (model_value - self._min) / (self._max - self._min)

    def num_parameters(self) -> int:
        return len(self.dist_args)

    def __eq__(self, other):
        return self.key == other.key


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
                 global_keys: Sequence[str], noise_keys: Sequence[str],
                 synapse_keys: Sequence[str], neuron_keys: Sequence[str],
                 nodes: List[int] = None, homogeneous_neurons: bool = False):
        """
        Interpretation order is 'global', 'noise', 'synapse', and 'neuron'.
        With 'neuron' being looped for how ever many neurons there are.

        :param graph: The graph used for the neural circuit optimization.
        :param translators: A list of translators with keys associated with
            all parameters that will be optimized.
        :param global_keys: A set of global keys.
        :param noise_keys: A set of keys associated with NEST noise models.
        :param synapse_keys: A set of keys associated with NEST synapse models.
        :param neuron_keys: A set of keys associated with NEST neuron models.
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
        self._translators = translators
        self._homogeneous_neurons = homogeneous_neurons
        self._global_keys = list(global_keys)
        self._noise_keys = list(noise_keys)
        self._synapse_keys = list(synapse_keys)
        self._neuron_keys = list(neuron_keys)

        self._global_parameters = {}
        self._noise_parameters = {}
        self._synapse_parameters = {}
        if self._homogeneous_neurons:
            self._neuron_parameters = {}
        else:
            self._neuron_parameters = {neuron_id: {} for neuron_id in self._nodes}

        if self._homogeneous_neurons:
            self._key_order = self._global_keys\
                              + self._noise_keys\
                              + self._synapse_keys\
                              + self._neuron_keys
        else:
            self._key_order = self._global_keys\
                              + self._noise_keys\
                              + self._synapse_keys\
                              + [key
                                 for _ in range(len(self._nodes))
                                 for key in self._neuron_keys]
        self._array_size = len(self._key_order)
        self._model_parameters = np.zeros(self._array_size)

    def __str__(self):
        s = "Translator\n"
        s += "="*7 + "Global" + "="*7 + "\n"
        s += str(self._global_parameters) + "\n"
        s += "="*7 + "Noise" + "="*7 + "\n"
        s += str(self._noise_parameters) + "\n"
        s += "="*7 + "Synapse" + "="*7 + "\n"
        s += str(self._synapse_parameters) + "\n"
        s += "="*7 + "Neurons" + "="*7 + "\n"
        if self._homogeneous_neurons:
            s += str(self._neuron_parameters) + "\n"
        else:
            for neuron_id, pars in self._neuron_parameters.items():
                s += "\tNEURON = " + str(neuron_id) + "\n"
                s += "\t" + str(pars) + "\n"
        return s

    @property
    def synapse_parameters(self) -> Dict[str, Any]:
        """
        :return: A dictionary of the synaptic parameters for the NEST model.
        """
        return self._synapse_parameters

    @property
    def neuron_parameters(self) -> Union[Dict[int, Dict[str, Any]], Dict[str, Any]]:
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
        if self._homogeneous_neurons:
            for key in self._neuron_keys:
                self._neuron_parameters[key] = self._model_parameters[c]
                c += 1
        else:
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
