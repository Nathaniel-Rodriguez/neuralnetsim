__all__ = ["ValueTranslator", "ArrayTranslator",
           "get_translator"]


import math
import networkx as nx
from typing import List
from typing import Set
from typing import Callable
from typing import Dict


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


class CompositeTranslator:
    """
    The CompositeTranslator allows for mapping multiple optimizer
    parameters to a single model parameter. This is desirable when some
    model parameters are coupled and it is easier to optimize over some
    composite parameter. For instance, voltage reset values can not exceed
    voltage threshold values, so instead of trying to couple the two parameters
    in some special case of the optimizer, it is easier to simply optimize
    over a value range and then translate the range back into the distinct
    voltage parameters.
    """
    def __init__(self, model_key: str, translator_list: List[ValueTranslator],
                 to_model_function: Callable[[...], float],
                 to_optimizer_function: Callable[[...], Dict]):
        """
        :param model_key: Key associated with the model parameter.
        :param translator_list: A list of optimizers to use for
        :param to_model_function: A function that takes keyword arguments for
        each optimizer parameter value required.
        :param to_optimizer_function: A function that takes a model parameter
        value associated with the model_key and a set of key-value pairs of
        other model parameter values and returns the associated
        non-composite parameter keys and values in a dictionary. This function
        is effectively the inverse of the to_optimizer_function. All model
        keywords will be dumped into this function.
        """
        self.model_key = model_key
        self._translator_list = translator_list
        self._translator_keys = set([translator.key
                                     for translator in self._translator_list])
        self._to_model_function = to_model_function
        self._to_optimizer_function = to_optimizer_function

    @property
    def translator_keys(self):
        """
        :return: Keys associated with all of the translators needed to
        contruct the composite model parameter.
        """
        return self._translator_keys

    @translator_keys.setter
    def translator_keys(self, v):
        raise NotImplementedError

    def to_model(self, **optimizer_kwargs) -> float:
        """
        Converts an optimizer value into a model value.
        :param optimizer_kwargs: key value pairs associated with each optimizer
        value that the translator needs. The keys property returns a set of
        key words that need to be filled. Raises a KeyError if not
        all key words are found or too many are given.
        :return: The converted value for the model.
        """
        if set(optimizer_kwargs.keys()) != self._translator_keys:
            raise KeyError("Key mismatch between required translator keys"
                           " and those provided. \nRequired: "
                           + str(self._translator_keys) + "\nFound: "
                           + str(set(optimizer_kwargs.keys())))
        function_kwargs = {key: get_translator(self._translator_list,
                                               key).to_model(value)
                           for key, value in optimizer_kwargs.items()}
        return self._to_model_function(**function_kwargs)

    def to_optimizer(self, **model_kwargs) -> Dict:
        """
        Converts a model value into a optimizer values.
        :param model_kwargs: Other model parameters that need to be used to
        determine the composite optimizer parameter value. Kwargs are not
        enforced. Method may use any or none of the provided keyword arguments.
        :return: The converted key value pairs for the optimizer.
        """
        return {key: get_translator(self._translator_list, key).to_optimizer(value)
                for key, value in self._to_optimizer_function(**model_kwargs).items()}


class ArrayTranslator:
    def __init__(self, graph: nx.DiGraph, translators: List[ValueTranslator],
                 neuron_keys: Set, synapse_keys: Set, noise_keys: Set,
                 global_keys: Set):
        self._graph = graph
        self._translators = translators
        self._neuron_keys = neuron_keys
        self._synapse_keys = synapse_keys
        self._noise_keys = noise_keys
        self._global_keys = global_keys
