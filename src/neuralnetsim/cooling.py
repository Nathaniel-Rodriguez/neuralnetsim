__all__ = ["CoolingSchedule", "ExponentialCoolingSchedule",
           "AdaptiveCoolingSchedule"]


import numpy as np
from math import inf
import math
from abc import ABC, abstractmethod


class CoolingSchedule(ABC):
    @abstractmethod
    def step(self, energy: float) -> float:
        """
        Increments the cooling schedule.
        :param energy: The current energy of the system.
        :return: The new temperature.
        """
        raise NotImplementedError


class ExponentialCoolingSchedule(CoolingSchedule):
    """
    Implements an exponential cooling scheduler for annealing algorithms.
    """
    def __init__(self, t0, cooling_factor, start: int = 0, stop: int = inf):
        """
        :param t0: Initial temperature.
        :param cooling_factor: Cooling factor for simulated annealing. Must be
        bound between [0,1]. A factor of 1.0 means no annealing takes place.
        :param start: The annealing step at which to begin cooling.
        :param stop: The annealing step at which to halt cooling.
        """
        if (cooling_factor < 0) or (cooling_factor > 1):
            raise AssertionError("Invalid input: Cooling factor must be"
                                 " between 0 and 1.")
        self.t0 = t0
        self.cooling_factor = cooling_factor
        self.start = start
        self.stop = stop
        self._t = t0
        self._step = 0

    def step(self, energy: None = float) -> float:
        """
        Increments the cooling schedule.
        :param energy: Unused for this scheduler.
        :return: The new temperature for this step.
        """
        if (self._step > self.start) and (self._step < self.stop):
            self._t = self.t0 * (self.cooling_factor ** (self._step - self.start))
        self._step += 1
        return self._t

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        raise NotImplementedError


class AdaptiveCoolingSchedule(CoolingSchedule):
    """
    Implements a cooling schedule based on the adaptive schedule developed by:
    Huang, M.D., Romeo, F., Sangiovanni-Vincentelli, A.L., 1986.
    An efficient general cooling schedule for simulated annealing,
    In: Proceedings of the IEEE International Conference on
    Computer-Aided Design, Santa Clara, pp. 381–384.
    This cooling schedule will slow down the cooling rate as the variance of
    the energy increases, and speed it up if the variance decreases.
    This cooling schedule is designed to work in environment where samples
    are accumulated asynchronously. When the algorithm is stepped it takes a
    temperature and energy of a sample and then makes an incremental weighted
    adjustment to the current temperature. Samples from temperatures very
    different from the current temperature, but close in time, are weighted
    low, while temperatures that are close to the current temperature are
    weighted highly. These energies are used to estimate the heat capacity
    of the system, which is then used by the referenced cooling schedule to
    update the temperature.
    """
    def __init__(self,
                 t0: float,
                 cooling_factor: float,
                 max_estimate_window: int = 10000,
                 decay_factor: float = 1.0,
                 hold_window: int = 100,
                 start: int = 0,
                 stop: int = inf,
                 tmin: float = 0.0):
        """
        :param t0: initial temperature
        :param cooling_factor: determines how quickly the cooling rate can
        change. A high value means fast change, while a low value means
        changes will be slow. Value is bound between [0,inf].
        :param max_estimate_window: the maximum allowed history to record.
        :param decay_factor: how strongly to penalize energy contributions from
        temperatures different from the current temperature when estimating
        the heat capacity. Larger values (>1) more strongly weigh distant
        temperatures, while smaller values (<1) more strongly weigh
        current temperatures.
        :param hold_window: how many initial steps to wait before updating the
        temperature. Since one sample is gained each step, this equates to the
        number of samples that will be used to generate the first heat
        capacity estimate.
        :param start: The annealing step at which to begin cooling.
        :param stop: The annealing step at which to halt cooling.
        :param tmin: the minimum allowed temperature. Default 0.0

        """
        self._t0 = t0  # initial temperature
        self._tc = self._t0  # current temperature
        self._start = start
        self._stop = stop
        self._g = cooling_factor
        self._tmin = tmin  # minimum temperature
        self._decay_factor = decay_factor
        self._t_record = [self._t0]  # log of annealed temperatures
        self._hold_window = hold_window  # how long to wait till t updates
        self._step_count = 0

        self._avg_e_history = []  # history of <e> for each change in t
        self._t_history = np.zeros(max_estimate_window)  # sample temp history
        self._e_history = np.zeros(max_estimate_window)  # sample energy history
        self._weights = np.zeros(max_estimate_window)  # sample weights
        self._var_buffer = np.zeros(max_estimate_window)  # pre-allocated buffer
        self._sample_index = max_estimate_window  # index for first valid sample

    @property
    def t(self):
        return self._tc

    @t.setter
    def t(self, value):
        raise NotImplementedError

    @property
    def record(self):
        return self._t_record

    @record.setter
    def record(self, value):
        raise NotImplementedError

    @property
    def ehistory(self):
        return self._e_history

    @ehistory.setter
    def ehistory(self, value):
        raise NotImplementedError

    @property
    def thistory(self):
        return self._t_history

    @thistory.setter
    def thistory(self, value):
        raise NotImplementedError

    def step(self, energy: float) -> float:
        if self._sample_index > 0:
            self._sample_index -= 1

        # pop the old sample and add the newest
        self._t_history[:-1] = self._t_history[1:]
        self._t_history[-1] = self._tc
        self._e_history[:-1] = self._e_history[1:]
        self._e_history[-1] = energy

        # wait to update the temperature until enough samples are accumulated
        # wait to update until after the start period
        # stop updating after the stop period
        if ((self._step_count < self._hold_window)
                or (self._step_count < self._start)):
            self._t_record.append(self._tc)
            tc = self._tc
        elif (self._tc >= self._tmin) and (self._step_count < self._stop):
            tc = self._update_temperature()
        else:
            self._t_record.append(self._tc)
            tc = self._tc
        self._step_count += 1
        return tc

    def _update_temperature(self) -> float:
        # update weights
        np.exp(
            np.multiply(
                self._decay_factor,
                np.subtract(self._tc, self._t_history, out=self._weights),
                out=self._weights),
            out=self._weights)

        # calculate energy standard deviation
        weighted_e_mean = np.average(self._e_history[self._sample_index:],
                                     weights=self._weights[self._sample_index:])
        if len(self._avg_e_history) == 0:
            self._avg_e_history.append(weighted_e_mean)
        self._avg_e_history.append(weighted_e_mean)
        weighted_e_std = math.sqrt(np.average(
            np.square(
                np.subtract(self._e_history[self._sample_index:],
                            weighted_e_mean,
                            out=self._var_buffer[self._sample_index:]),
                out=self._var_buffer[self._sample_index:]),
            weights=self._weights[self._sample_index:]
        ) / self._weights[self._sample_index:].sum())

        # update temperature
        if not math.isclose(weighted_e_std, 0.0, abs_tol=1e-15):
            self._tc = self._tc * math.exp(-self._g * self._tc / weighted_e_std)
        else:
            self._tc = self._tc
        if self._tc < self._tmin:
            self._tc = self._tmin
        self._t_record.append(self._tc)
        return self._tc
