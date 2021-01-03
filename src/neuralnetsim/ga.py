__all__ = ["AdaptiveCoolingSchedule",
           "ACSGa"]


import math
import numpy as np
from typing import Callable
from typing import Tuple
from typing import Any
from time import sleep
from distributed import Client
from distributed import as_completed


class AdaptiveCoolingSchedule:
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
                 tmin: float = 0.0,
                 max_estimate_window: int = 10000,
                 decay_factor: float = 1.0,
                 hold_window: int = 100):
        """
        :param t0: initial temperature
        :param cooling_factor: determines how quickly the cooling rate can
        change. A high value means fast change, while a low value means
        changes will be slow. Value is bound between [0,inf]. Units of energy
        per temperature.
        :param tmin: the minimum allowed temperature. Default 0.0
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
        """

        self._t0 = t0  # initial temperature
        self._tc = self._t0  # current temperature
        self._g = cooling_factor
        self._tmin = tmin  # minimum temperature
        self._decay_factor = decay_factor
        self._t_log = [self._t0]  # log of annealed temperatures
        self._hold_window = hold_window  # how long to wait till t updates
        self._step_count = 0

        self._avg_e_history = []  # history of <e> for each change in t
        self._t_history = np.zeros(max_estimate_window)  # sample temp history
        self._e_history = np.zeros(max_estimate_window)  # sample energy history
        self._weights = np.zeros(max_estimate_window)  # sample weights
        self._var_buffer = np.zeros(max_estimate_window)  # pre-allocated buffer
        self._sample_index = max_estimate_window  # index for first valid sample

    @property
    def temperature(self):
        return self._tc

    @temperature.setter
    def temperature(self, value):
        raise NotImplementedError

    @property
    def log(self):
        return self._t_log

    @log.setter
    def log(self, value):
        raise NotImplementedError

    def step(self, sample_t: float, sample_e: float) -> float:
        """
        Steps the cooling schedule.
        :param sample_t: temperature of the given sample
        :param sample_e: energy of the given sample
        :return: the new temperature
        """
        if self._sample_index > 0:
            self._sample_index -= 1

        # pop the old sample and add the newest
        self._t_history[:-1] = self._t_history[1:]
        self._t_history[-1] = sample_t
        self._e_history[:-1] = self._e_history[1:]
        self._e_history[-1] = sample_e

        # wait to update the temperature until enough samples are accumulated
        if self._step_count < self._hold_window:
            self._step_count += 1
            self._t_log.append(self._tc)
            return self._tc
        elif self._tc >= self._tmin:
            return self._update_temperature()
        else:
            self._t_log.append(self._tc)
            return self._tc

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
        self._t_log.append(self._tc)
        return self._tc


class CostWorker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def initialize_worker(*args, **kwargs) -> CostWorker:
    return CostWorker(*args, **kwargs)


def dispatch_work(cost_function: Callable[[np.ndarray, Any], float],
                  x: np.ndarray,
                  worker: CostWorker,
                  temperature: float,
                  worker_id: int) -> Tuple[float, np.ndarray, float, int]:
    return cost_function(x, *worker.args, **worker.kwargs), x, temperature, worker_id


class ACSGa:
    """
    """
    def __init__(
            self,
            x0: np.ndarray,
            population_size: int,
            seed: int,
            cooling_schedule: AdaptiveCoolingSchedule,
            dtype=np.float32):
        self._x0 = x0.astype(dtype=dtype)
        self._population_size = population_size
        self._seed = seed
        self._ndim = len(self._x0)
        self._rng = np.random.RandomState(self._seed)
        self._step = 0
        self._sigma0 = cooling_schedule.temperature
        self._sigma = cooling_schedule.temperature
        self._cooling_schedule = cooling_schedule
        self.cost_history = []

        self._cost_rank_sum = self._population_size \
                              * (self._population_size + 1) / 2
        self._selection_probabilities = \
            [(self._population_size - i) / self._cost_rank_sum
                     for i in range(self._population_size)]

        self._population = np.zeros((self._population_size, self._ndim), dtype=dtype)
        self._costs = np.zeros(self._population_size, dtype=dtype)
        self._costs[:] = np.inf

    def get_best(self) -> np.ndarray:
        return np.copy(self._population[np.argmin(self._costs), :]).flatten()

    def run(
            self,
            cost_function: Callable[[np.ndarray, Any], float],
            client: Client,
            num_iterations,
            **kwargs
    ):
        if kwargs is None:
            kwargs = {}

        # setup evo strat workers
        sleep(5)  # wait on workers to connect
        num_workers = len(client.scheduler_info()['workers'])
        if num_workers == 0:
            raise ValueError("Error: there are no workers.")
        dask_workers = list(client.scheduler_info()['workers'].keys())
        if len(dask_workers) == 0:
            raise ValueError("Error: there are no workers.")
        evo_workers = [client.submit(initialize_worker,
                                     **kwargs,
                                     workers=[worker],
                                     pure=False)
                       for worker in dask_workers]

        # submit jobs to all workers or till num iterations is saturated
        jobs = []
        for worker_id in range(min(num_iterations, len(evo_workers))):
            jobs.append(client.submit(
                dispatch_work,
                cost_function,
                self._mutation(self._crossover(self._selection(),
                                               self._selection())),
                evo_workers[worker_id],
                self._sigma,
                worker_id,
                workers=[dask_workers[worker_id]],
                pure=False))
            self._step += 1

        # iterate ga until num_iterations reached
        working_batch = as_completed(jobs)
        for completed_job in working_batch:
            cost, x, temperature, worker_id = completed_job.result()
            self._anneal(cost, temperature)
            self._replacement(x, cost)
            self._update_cost_history()

            if self._step < num_iterations:
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    self._mutation(self._crossover(self._selection(),
                                                   self._selection())),
                    evo_workers[worker_id],
                    self._sigma,
                    worker_id,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1

    def _selection(self) -> np.ndarray:
        return np.copy(self._population[self._rng.choice(
            np.argsort(self._costs),
            size=1,
            replace=False,
            p=self._selection_probabilities), :]).flatten()

    def _mutation(self, x: np.ndarray) -> np.ndarray:
        x += self._rng.normal(scale=self._sigma, size=self._ndim)
        return x

    def _crossover(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        swap_mask = self._rng.randint(2, size=self._ndim)
        x[swap_mask] = y[swap_mask]
        return x

    def _replacement(self, x: np.ndarray, cost: float):
        # find lowest fitness pop and then replace
        weakest_member_index = None
        highest_cost = -np.inf
        for i in range(self._population_size):
            if self._costs[i] > highest_cost:
                weakest_member_index = i
                highest_cost = self._costs[i]

        if cost < highest_cost and weakest_member_index is not None:
            self._population[weakest_member_index, :] = x[:]
            self._costs[weakest_member_index] = cost

    def _anneal(self, sample_cost, sample_temperature):
        self._sigma = self._cooling_schedule.step(
            sample_temperature, sample_cost)

    def _update_cost_history(self):
        self.cost_history.append(
            [cost
             for cost in self._costs
             if cost != np.inf])
