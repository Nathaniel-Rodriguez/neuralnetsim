__all__ = ["MapGa",
           "MapExploit",
           "MapExploreExploit",
           "DrawMember"]


import math
import statistics
import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import Callable
from typing import Tuple
from typing import Any
from typing import List
from time import sleep
from distributed import Client
from distributed import as_completed
from collections import deque
from sklearn.neighbors import NearestNeighbors
from neuralnetsim.cooling import AdaptiveCoolingSchedule


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
            num_opt_steps: int = 1
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

    def get_temperature_logs(self):
        return [schedule.log
                for ci in self._cell_cooling_schedules
                for schedule in ci]

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
                    i,
                    j,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1

    def _anneal(self, i, j, sample_cost, sample_temperature):
        self._cell_cooling_schedules[i][j].async_step(
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


class MapExploit:
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
        self._cell_updates = deque()

    def _find_top_n(self, n) -> List[Tuple[int, int]]:
        top_n = []
        for _ in range(n):
            lowest_cost = math.inf
            i_min = 0
            j_min = 0
            for i in range(self._num_f1_cells):
                for j in range(self._num_f2_cells):
                    if lowest_cost > self._cell_cost[i][j] and ((i, j) not in top_n):
                        lowest_cost = self._cell_cost[i][j]
                        i_min = i
                        j_min = j
            top_n.append((i_min, j_min))
        return top_n

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

    def get_temperature_logs(self):
        return [schedule.log
                for ci in self._cell_cooling_schedules
                for schedule in ci]

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

            # initial solution setup
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
            # run from cells
            elif self._step < num_iterations:
                chosen, i, j = self._select()
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    chosen,
                    map_workers[worker_id],
                    self._cell_cooling_schedules[i][j].temperature,
                    worker_id,
                    i,
                    j,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1

    def _anneal(self, i, j, sample_cost, sample_temperature):
        self._cell_cooling_schedules[i][j].async_step(
            sample_temperature, sample_cost)

    def _replacement(self, x: np.ndarray,
                     result: Tuple[float, float, float]):
        i, j = self._get_cell(result[1], result[2])
        if result[0] < self._cell_cost[i][j]:
            self._cell_member[i][j] = x
            self._cell_cost[i][j] = result[0]
            # update cell log
            self._cell_updates.append((i, j))
            if len(self._cell_updates) > 5:
                self._cell_updates.popleft()

    def _get_cell(self, f1, f2) -> Tuple[int, int]:
        return np.searchsorted(self._feature1_cells, f1), \
               np.searchsorted(self._feature2_cells, f2)

    def _select(self) -> Tuple[np.ndarray, int, int]:
        x = None
        if self._rng.random() > 0.5:
            i, j = self._find_top_n(2)[self._rng.randint(0, 2)]
            x = self._cell_member[i][j]
        else:
            i, j = self._cell_updates[self._rng.randint(0, len(self._cell_updates))]
            x = self._cell_member[i][j]
        x = np.copy(x)
        x += self._rng.normal(
            scale=self._cell_cooling_schedules[i][j].temperature,
            size=self._ndim)
        return x, i, j


class DrawMember:
    @abstractmethod
    def draw(self, rng: np.random.RandomState) -> np.ndarray:
        """

        :param rng: Source for random numbers.
        :return: A member of the population.
        """
        raise NotImplementedError


class DefaultMemberSource(DrawMember):
    def __init__(self, ndim):
        self._ndim = ndim

    def draw(self, rng: np.random.RandomState):
        return rng.uniform(0.0, 1.0, size=self._ndim)


class MapExploreExploit:
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
            k_nearest: int = 10,
            initial_member_source: DrawMember = None
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
        self._cell_novelty = [[0.0
                               for _ in range(self._num_f2_cells)]
                              for _ in range(self._num_f1_cells)]
        self._step = 0
        self._rng = np.random.RandomState(self._seed)
        if initial_member_source is None:
            self._member_source = DefaultMemberSource(self._ndim)
        else:
            self._member_source = initial_member_source
        self._sigma0 = cooling_schedule.temperature
        self._cell_cooling_schedules = [[copy.deepcopy(cooling_schedule)
                                         for _ in range(self._num_f2_cells)]
                                        for _ in range(self._num_f1_cells)]
        self._cell_member = [[None
                              for _ in range(self._num_f2_cells)]
                             for _ in range(self._num_f1_cells)]
        self._cell_bc = [[None
                              for _ in range(self._num_f2_cells)]
                             for _ in range(self._num_f1_cells)]
        self._cell_updates = deque()
        self._behavior_archive = []
        self._k_nearest = k_nearest
        self._neighbors = NearestNeighbors(
            n_neighbors=self._k_nearest,
            algorithm='ball_tree',
            metric='euclidean')

    def _find_top_cost(self, n) -> List[Tuple[int, int]]:
        top_n = []
        for _ in range(n):
            lowest_cost = math.inf
            i_min = 0
            j_min = 0
            for i in range(self._num_f1_cells):
                for j in range(self._num_f2_cells):
                    if (lowest_cost > self._cell_cost[i][j])\
                            and ((i, j) not in top_n)\
                            and (self._cell_member[i][j] is not None):
                        lowest_cost = self._cell_cost[i][j]
                        i_min = i
                        j_min = j
            top_n.append((i_min, j_min))
        return top_n

    def get_best_n(self, n) -> List[np.ndarray]:
        return [self._cell_member[i][j] for i, j in self._find_top_cost(n)]

    def _find_top_novel(self, n) -> List[Tuple[int, int]]:
        top_n = []
        for _ in range(n):
            greatest_novelty = 0.0
            i_min = 0
            j_min = 0
            for i in range(self._num_f1_cells):
                for j in range(self._num_f2_cells):
                    if greatest_novelty < self._cell_novelty[i][j]\
                            and ((i, j) not in top_n)\
                            and (self._cell_member[i][j] is not None):
                        greatest_novelty = self._cell_novelty[i][j]
                        i_min = i
                        j_min = j
            top_n.append((i_min, j_min))
        return top_n

    def _find_top_novel_inbound(self, n) -> List[Tuple[int, int]]:
        top_n = []
        for _ in range(n):
            greatest_novelty = 0.0
            i_min = 0
            j_min = 0
            for i in range(self._num_f1_cells - 1):
                for j in range(self._num_f2_cells - 1):
                    if greatest_novelty < self._cell_novelty[i][j]\
                            and ((i, j) not in top_n)\
                            and (self._cell_member[i][j] is not None):
                        greatest_novelty = self._cell_novelty[i][j]
                        i_min = i
                        j_min = j
            top_n.append((i_min, j_min))
        return top_n

    def _get_novelty(self, f1, f2):
        closest = []
        for a1, a2 in self._behavior_archive:
            closest.append(math.sqrt((f2-a2)**2 + (f1-a1)**2))
        if len(closest) == 0:
            return 0.0
        return statistics.mean(sorted(closest)[:min(self._k_nearest, len(closest))])

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

    def get_novelty_map(self) -> np.ndarray:
        self._update_cell_novelties()
        return np.array(self._cell_novelty)

    def get_cost_map(self) -> np.ndarray:
        return np.array(self._cell_cost)

    def get_temperature_logs(self):
        return [schedule.log
                for ci in self._cell_cooling_schedules
                for schedule in ci]

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
                self._member_source.draw(self._rng),
                map_workers[worker_id],
                self._sigma0,
                worker_id,
                workers=[dask_workers[worker_id]],
                pure=False))
            self._num_initial_solutions -= 1
            # self._step += 1

        # iterate map until num_iterations reached
        working_batch = as_completed(jobs)
        for completed_job in working_batch:
            result, x, temperature, worker_id, i, j = completed_job.result()
            if (i is not None) and (j is not None):
                self._anneal(i, j, result[0], temperature)
            self._replacement(x, result)
            if (result[1], result[2]) not in self._behavior_archive:
                self._behavior_archive.append((result[1], result[2]))

            # initial solution setup
            if self._num_initial_solutions > 0:
                self._num_initial_solutions -= 1
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    self._member_source.draw(self._rng),
                    map_workers[worker_id],
                    self._sigma0,
                    worker_id,
                    workers=[dask_workers[worker_id]],
                    pure=False))
            # run exploitation
            elif self._step < num_iterations and (self._rng.random() < 0.5):
                chosen, i, j = self._exploit_select()
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    chosen,
                    map_workers[worker_id],
                    self._cell_cooling_schedules[i][j].temperature,
                    worker_id,
                    i,
                    j,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1
            # run exploration
            elif self._step < num_iterations:
                chosen, i, j = self._explore_select()
                working_batch.add(client.submit(
                    dispatch_work,
                    cost_function,
                    chosen,
                    map_workers[worker_id],
                    self._cell_cooling_schedules[i][j].temperature,
                    worker_id,
                    i,
                    j,
                    workers=[dask_workers[worker_id]],
                    pure=False))
                self._step += 1

            # updated cell novelties every 100 steps
            if self._step % 100 == 0:
                self._update_cell_novelties()

    def _update_cell_novelties(self):
        self._neighbors.fit(self._behavior_archive)
        for c1 in range(self._num_f1_cells):
            for c2 in range(self._num_f2_cells):
                if self._cell_bc[c1][c2] is not None:
                    d, _ = self._neighbors.kneighbors(
                        [self._cell_bc[c1][c2]],
                        n_neighbors=min(len(self._behavior_archive),
                                        self._k_nearest+1))  # including self
                    self._cell_novelty[c1][c2] = d[0, 1:].mean()  # exclude self

    def _anneal(self, i, j, sample_cost, sample_temperature):
        self._cell_cooling_schedules[i][j].async_step(
            sample_temperature, sample_cost)

    def _replacement(self, x: np.ndarray,
                     result: Tuple[float, float, float]):
        i, j = self._get_cell(result[1], result[2])
        novelty = self._get_novelty(result[1], result[2])
        if result[0] < self._cell_cost[i][j]:
            self._cell_member[i][j] = x
            self._cell_cost[i][j] = result[0]
            self._cell_novelty[i][j] = novelty
            self._cell_bc[i][j] = (result[1], result[2])
            # update cell log
            self._cell_updates.append((i, j))
            if len(self._cell_updates) > 5:
                self._cell_updates.popleft()

    def _get_cell(self, f1, f2) -> Tuple[int, int]:
        return np.searchsorted(self._feature1_cells, f1), \
               np.searchsorted(self._feature2_cells, f2)

    def _exploit_select(self) -> Tuple[np.ndarray, int, int]:
        x = None
        if self._rng.random() > 0.5:
            top = self._find_top_cost(2)
            i, j = top[self._rng.randint(0, max(len(top), 2))]
            x = self._cell_member[i][j]
        else:
            i, j = self._cell_updates[self._rng.randint(0, len(self._cell_updates))]
            x = self._cell_member[i][j]
        x = np.copy(x)
        x += self._rng.normal(
            scale=self._cell_cooling_schedules[i][j].temperature,
            size=self._ndim)
        return x, i, j

    def _explore_select(self) -> Tuple[np.ndarray, int, int]:
        top = self._find_top_novel_inbound(5)
        i, j = top[self._rng.randint(0, max(len(top), 5))]
        x = np.copy(self._cell_member[i][j])
        x += self._rng.normal(
            scale=self._cell_cooling_schedules[i][j].temperature,
            size=self._ndim)
        return x, i, j
