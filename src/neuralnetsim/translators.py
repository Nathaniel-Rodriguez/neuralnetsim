import math


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


class ArrayTranslator:
    def __init__(self):
        pass
