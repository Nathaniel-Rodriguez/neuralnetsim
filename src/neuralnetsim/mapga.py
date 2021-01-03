__all__ = ["AdaptiveCoolingSchedule",
           "MapGa"]


import math
import numpy as np
import copy
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


def dispatch_work(cost_function: Callable[[np.ndarray, Any], Tuple[float, float, float]],
                  x: np.ndarray,
                  worker: CostWorker,
                  temperature: float,
                  worker_id: int,
                  cell_i: int = None,
                  cell_j: int = None) -> Tuple[float, np.ndarray, float, int]:
    return cost_function(x, *worker.args, **worker.kwargs), x, temperature, worker_id, cell_i, cell_j


class MapGa:
    """
    """
    def __init__(
            self,
            feature1_cells: np.ndarray,
            feature2_cells: np.ndarray,
            num_initial_solutions: int,
            ndim: int,
            seed: int,
            cooling_schedule: AdaptiveCoolingSchedule,
    ):
        # feature cells, basically the bins
        self._feature1_cells = feature1_cells
        self._feature2_cells = feature2_cells
        # there is 1 more cell than bins to cover values greater than the last bin
        self._num_f1_cells = len(self._feature1_cells) + 1
        self._num_f2_cells = len(self._feature2_cells) + 1
        self._ndim = ndim
        self._num_initial_solutions = num_initial_solutions
        self._seed = seed
        self._cell_cost = [[math.inf
                            for _ in range(self._num_f2_cells)]
                           for _ in range(self._num_f1_cells)]
        self._step = 0
        self._rng = np.random.RandomState(self._seed)
        self._sigma0 = cooling_schedule.temperature
        self._cell_cooling_schedules = [[copy.deepcopy(cooling_schedule)
                                         for _ in range(self._num_f2_cells)]
                                        for _ in range(self._num_f1_cells)]
        self._cell_member = [[None
                              for _ in range(self._num_f2_cells)]
                             for _ in range(self._num_f1_cells)]

    def get_best(self) -> np.ndarray:
        lowest_cost = math.inf
        i_min = 0
        j_min = 0
        for i in range(self._num_f1_cells):
            for j in range(self._num_f2_cells):
                if lowest_cost > self._cell_cost[i][j]:
                    lowest_cost = self._cell_cost[i][j]
                    i_min = i
                    j_min = j
        return self._cell_member[i_min][j_min]

    def get_cost_map(self) -> np.ndarray:
        return np.array(self._cell_cost)

    def run(
            self,
            cost_function: Callable[[np.ndarray, Any], Tuple[float, float, float]],
            client: Client,
            num_iterations,
            **kwargs
    ):
        if kwargs is None:
            kwargs = {}

        # setup workers
        sleep(5)  # wait on workers to connect
        num_workers = len(client.scheduler_info()['workers'])
        if num_workers == 0:
            raise ValueError("Error: there are no workers.")
        dask_workers = list(client.scheduler_info()['workers'].keys())
        if len(dask_workers) == 0:
            raise ValueError("Error: there are no workers.")
        map_workers = [client.submit(initialize_worker,
                                     **kwargs,
                                     workers=[worker],
                                     pure=False)
                       for worker in dask_workers]

        # submit jobs to all workers for initial randomized batch
        jobs = []
        for worker_id in range(len(map_workers)):
            jobs.append(client.submit(
                dispatch_work,
                cost_function,
                self._rng.uniform(0.0, 1.0, size=self._ndim),
                map_workers[worker_id],
                self._sigma0,
                worker_id,
                workers=[dask_workers[worker_id]],
                pure=False))
            self._step += 1
            self._num_initial_solutions -= 1

        # iterate map until num_iterations reached
        working_batch = as_completed(jobs)
        for completed_job in working_batch:
            result, x, temperature, worker_id, i, j = completed_job.result()
            if (i is not None) and (j is not None):
                self._anneal(i, j, result[0], temperature)
            self._replacement(x, result)

            if self._num_initial_solutions > 0:
                self._num_initial_solutions -= 1
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    self._rng.uniform(0.0, 1.0, size=self._ndim),
                    map_workers[worker_id],
                    self._sigma0,
                    worker_id,
                    workers=[dask_workers[worker_id]],
                    pure=False))

            elif self._step < num_iterations:
                chosen, i, j = self._select()
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    chosen,
                    map_workers[worker_id],
                    self._cell_cooling_schedules[i][j].temperature,
                    worker_id,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1

    def _anneal(self, i, j, sample_cost, sample_temperature):
        self._cell_cooling_schedules[i][j].step(
            sample_temperature, sample_cost)

    def _replacement(self, x: np.ndarray,
                     result: Tuple[float, float, float]):
        i, j = self._get_cell(result[1], result[2])
        if result[0] < self._cell_cost[i][j]:
            self._cell_member[i][j] = x
            self._cell_cost[i][j] = result[0]

    def _get_cell(self, f1, f2) -> Tuple[int, int]:
        return np.searchsorted(self._feature1_cells, f1),\
               np.searchsorted(self._feature2_cells, f2)

    def _select(self) -> Tuple[np.ndarray, int, int]:
        x = None
        while x is None:
            i = self._rng.randint(self._num_f1_cells)
            j = self._rng.randint(self._num_f2_cells)
            x = self._cell_member[i][j]
        x = np.copy(x)
        x += self._rng.normal(
            scale=self._cell_cooling_schedules[i][j].temperature,
            size=self._ndim)
        return x, i, j
