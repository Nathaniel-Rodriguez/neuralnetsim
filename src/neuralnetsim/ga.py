__all__ = ["ACSGa"]


import math
import numpy as np
from typing import Callable
from typing import Tuple
from typing import Any
from time import sleep
from distributed import Client
from distributed import as_completed
from neuralnetsim.cooling import AdaptiveCoolingSchedule


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
        self._sigma = self._cooling_schedule.async_step(
            sample_temperature, sample_cost)

    def _update_cost_history(self):
        self.cost_history.append(
            [cost
             for cost in self._costs
             if cost != np.inf])
