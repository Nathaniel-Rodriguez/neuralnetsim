__all__ = ["SCMAEvoStrat"]


import math
import numpy as np
import copy
from pathlib import Path
from typing import Dict
from typing import Callable
from typing import List
from typing import Tuple
from typing import Any
from time import sleep
from distributed import Client
from distributed import as_completed
from neuralnetsim.sliceops import *
from neuralnetsim import save


class EvoStratState:
    """
    EvoStratState manages the internal state of the evolutionary strategy
    and its update methods.
    """
    def __init__(self,
                 x0: np.ndarray,
                 sigma0: float,
                 seed: int = None,
                 population_size: int = None,
                 mu: int = None,
                 s_dim: float = None,
                 mu_eff: float = None,
                 c_c: float = None,
                 c_s: float = None,
                 c_1: float = None,
                 c_mu: float = None,
                 d_s: float = None,
                 rnd_table_size: int = 2000000,
                 max_table_step: int = 5,
                 dtype=np.float32):
        """
        :param x0: Initial centroid.
        :param sigma0: Initial step-size.
        :param seed: Random number seed.
        :param population_size: Size of the evolutionary population [defaulted].
        :param mu: Number of parents [defaulted].
        :param s_dim: Scaling factor for sigma based on number of dimensions [defaulted].
        :param mu_eff: Variance effective selective mass [defaulted].
        :param c_c: Covariance matrix learning rate [defaulted].
        :param c_s: Sigma learning rate [defaulted].
        :param c_1: First-rank covariance learning rate [defaulted].
        :param c_mu: Mu-rank covariance learning rate [defaulted].
        :param d_s: Sigma damping constant [defaulted].
        :param rnd_table_size: Size of random number table [defaulted].
        :param max_table_step: Maximum random step-size for controlling level
        of randomness [defaulted].
        :param dtype: Numpy data type for arrays (default: np.float32).
        """
        self._dtype = dtype
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._table = self._rng.randn(rnd_table_size)
        self._max_table_step = max_table_step
        self._x = x0.copy().astype(dtype)
        self._x_last = x0.copy().astype(dtype)
        self._x_buffer = x0.copy().astype(dtype)
        self._ndim = len(x0)
        self._ndim_buff = np.zeros(self._ndim, dtype=dtype)
        self._ndim_buff2 = np.zeros(self._ndim, dtype=dtype)
        self._step = 0
        self._c_diff = np.zeros(self._ndim, dtype=dtype)
        self._lambda = 4 + math.floor(3 * math.log(self._ndim))\
            if population_size is None else population_size
        self._population = np.zeros((self._lambda, self._ndim), dtype=dtype)
        self._mutations = np.zeros((self._lambda, self._ndim), dtype=dtype)
        self._mu = int(self._lambda / 2.) if mu is None else mu
        self._s = sigma0
        self._s_dim = math.sqrt(self._ndim)\
                      * (1 - 1. / (4. * self._ndim)
                         + 1. / (21. * self._ndim ** 2))\
            if s_dim is None else s_dim
        self._c_mat = np.ones(self._ndim, dtype=dtype)
        self._c_mat_path = np.zeros(self._ndim, dtype=dtype)
        self._ordered_parents = np.zeros((self._mu, self._ndim), dtype=dtype)
        self._ordered_parents_buff = np.zeros((self._mu, self._ndim), dtype=dtype)
        self._ordered_parents_tbuff = np.zeros((self._ndim, self._mu), dtype=dtype)
        self._s_path = np.zeros(self._ndim, dtype=dtype)
        self._weights = math.log(self._mu + 0.5)\
                        - np.log(np.arange(1, self._mu + 1))
        self._weights /= sum(self._weights)
        self._weights = self._weights.astype(dtype, copy=True)
        self._mu_eff = 1. / sum(self._weights ** 2) if mu_eff is None else mu_eff
        self._c_c = 4. / (self._ndim + 4) if c_c is None else c_c
        self._c_s = (self._mu_eff + 2) / (self._ndim + self._mu_eff + 3) \
            if c_s is None else c_s
        self._c_1 = 2. / ((self._ndim + 1.3)**2 + self._mu_eff)\
                    * (self._ndim + 2) / 3 \
            if c_1 is None else c_1
        self._c_mu = min([1 - self._c_1,
                          2. * (self._mu_eff - 2 + 1 / self._mu_eff)
                          / ((self._ndim + 2)**2 + self._mu_eff)]) \
            if c_mu is None else c_mu
        self._d_s = 1 + 2 * max(0., math.sqrt((self._mu_eff - 1)
                                              / (self._ndim + 1)) - 1) + self._c_s \
            if d_s is None else d_s

        self._d_diag = self._c_mat**0.5

    @property
    def centroid(self):
        return self._x

    @property
    def population(self):
        return self._population

    @property
    def population_size(self):
        return self._lambda

    @property
    def sigma(self):
        return self._s

    def update_population(self) -> int:
        for i in range(self._lambda):
            self._population[i, :] = self._x[:]
            param_slices = random_slices(self._rng, self._ndim, self._ndim, 1)
            table_slices = random_slices(self._rng, len(self._table),
                                         self._ndim, self._max_table_step)
            param_slices, table_slices = match_slices(param_slices, table_slices)
            multi_slice_assign(self._mutations[i], self._table,
                               param_slices, table_slices)
        np.multiply(self._mutations, self._d_diag, out=self._mutations)
        np.multiply(self._mutations, self._s, out=self._mutations)
        np.add(self._population, self._mutations, out=self._population)
        return self._step

    def update_centroid(self, costs: List[float]) -> int:
        # sort costs and then add weighted parents to centroid
        sorted_parent_indices = np.argsort(costs)
        self._x_last[:] = self._x[:]
        for i, parent_index in enumerate(sorted_parent_indices[:self._mu]):
            self._ordered_parents[i, :] = self._population[parent_index, :]
        np.dot(self._weights, self._ordered_parents, out=self._x)

        # calculate centroid movement and hsig for path and matrix update sequence
        self._c_diff[:] = self._x[:]
        np.subtract(self._c_diff, self._x_last, out=self._c_diff)
        hsig = float((np.linalg.norm(self._s_path) /
                      math.sqrt(1. - (1. - self._c_s)
                                ** (2. * (self._step + 1.))) / self._s_dim
                      < (1.4 + 2. / (self._ndim + 1.))))
        self._step += 1

        # update sigma path
        np.multiply(self._s_path, 1 - self._c_s, out=self._s_path)
        self._ndim_buff[:] = self._c_diff[:]
        np.divide(self._ndim_buff, self._d_diag, out=self._ndim_buff)
        np.multiply(self._ndim_buff,
                    math.sqrt(self._c_s * (2 - self._c_s) * self._mu_eff) / self._s,
                    out=self._ndim_buff)
        np.add(self._s_path, self._ndim_buff, out=self._s_path)

        # update covariance matrix path
        np.multiply(self._c_mat_path, 1 - self._c_c, out=self._c_mat_path)
        self._ndim_buff[:] = self._c_diff[:]
        np.multiply(self._ndim_buff,
                    hsig * math.sqrt(self._c_c * (2 - self._c_c) * self._mu_eff) / self._s,
                    out=self._ndim_buff)
        np.add(self._c_mat_path, self._ndim_buff, out=self._c_mat_path)

        # update covariance matrix
        np.multiply(self._c_mat,
                    1 - self._c_1 - self._c_mu + (1 - hsig)
                    * self._c_1 * self._c_c * (2 - self._c_c),
                    out=self._c_mat)
        self._ndim_buff[:] = self._c_mat_path[:]
        np.multiply(self._ndim_buff, self._c_mat_path, out=self._ndim_buff)
        np.multiply(self._ndim_buff, self._c_1, out=self._ndim_buff)
        np.subtract(self._ordered_parents, self._x_last, out=self._ordered_parents)
        self._ordered_parents_buff[:] = self._ordered_parents[:]
        np.multiply(self._weights, self._ordered_parents.T, out=self._ordered_parents_tbuff)
        np.multiply(self._ordered_parents_tbuff, self._ordered_parents.T, out=self._ordered_parents_tbuff)
        np.sum(self._ordered_parents_tbuff, axis=-1, out=self._ndim_buff2)
        np.multiply(self._ndim_buff2 / self.sigma ** 2, self._c_mu, out=self._ndim_buff2)
        np.add(self._c_mat, self._ndim_buff, out=self._c_mat)
        np.add(self._c_mat, self._ndim_buff2, out=self._c_mat)

        # update sigma
        self._s *= np.exp((np.linalg.norm(self._s_path)
                           / self._s_dim - 1.) * self._c_s / self._d_s)

        # update diagonal
        np.sqrt(self._c_mat, out=self._d_diag)
        return self._step


class EvoStratWorker:
    """
    EvoStratWorker maintains and advances the evolutionary strategy state on
    Dask workers.
    """
    def __init__(self, cost_kwargs: Dict = None, *args, **kwargs):
        """
        :param cost_kwargs: Any heavy arguments for the cost function
        that should only be initialized once. Will be stored here and will be
        forwarded to the cost function on call.
        :param args: Arguments for the EvoStratState.
        :param kwargs: Arguments for the EvoStratState.
        """
        self._state = EvoStratState(*args, **kwargs)
        self.cost_kwargs = cost_kwargs

    def retrieve_pop(self, pop_id: int) -> np.ndarray:
        return self._state.population[pop_id]

    def update_population(self) -> int:
        return self._state.update_population()

    def update_centroid(self, costs: List[float]) -> int:
        return self._state.update_centroid(costs)


def initialize_worker(*args, **kwargs) -> EvoStratWorker:
    """
    Initialized the EvoStratWorker on a Dask client.
    """
    return EvoStratWorker(*args, **kwargs)


def update_pops(worker: EvoStratWorker):
    return worker.update_population()


def scatter_cost(costs: List[float], worker: EvoStratWorker):
    return worker.update_centroid(costs)


def dispatch_work(cost_function: Callable[[np.ndarray, Any], float],
                  worker: EvoStratWorker,
                  pop_id: int,
                  worker_id: int) -> Tuple[float, int, int]:
    """
    Dispatches a new population evaluation workload to workers.
    :param cost_function: A function that evaluates the cost of a given agent.
    :param worker: The evolutionary strategy's worker.
    :param pop_id: ID for the member of the population to be evaluated.
    :param worker_id: ID for the Dask worker.
    :return: Cost evaluated for a given member of the population.
    """
    return cost_function(worker.retrieve_pop(pop_id),
                         **worker.cost_kwargs), pop_id, worker_id


class SCMAEvoStrat:
    """
    Implements a separable covariance-matrix adaptation evolutionary strategy.
    """
    def __init__(self, **kwargs):
        """
        :param kwargs: Arguments to forward to EvoStratState.
        """
        self._kwargs = kwargs.copy()
        self._state = EvoStratState(**kwargs)
        self.generation_history = []
        self.cost_history = []
        self.centroid_history = []
        self.sigma_history = []
        self.sigma_path_history = []
        self.cov_path_history = []

    def to_file(self, filename: Path):
        """
        Writes out the history of the ES to file.
        :param filename: Save file path.
        """
        save(
            {'cost_history': self.cost_history,
             'centroid_history': self.centroid_history,
             'sigma_history': self.sigma_history,
             'sigma_path_history': self.sigma_path_history,
             'cov_path_history': self.cov_path_history,
             'generation_history': self.generation_history},
            filename
        )

    def run(self, cost_function: Callable[[np.ndarray, Any], float],
            client: Client, num_iterations, cost_kwargs: Dict = None,
            enable_path_history: bool = False):
        """
        Initiates the ES run.
        :param cost_function: A callable function that will evaluate the cost
        of a given array of parameters.
        :param client: A Dask client for distributing execution.
        :param num_iterations: How many evolutionary steps to take.
        :param cost_kwargs: Arguments to save on the worker which will be
        forwarded to the cost function at execution. These can be large
        data sets or objects, as they are only passed to the worker once.
        :param enable_path_history: Whether to save paths to the history.
        :return: None
        """
        if cost_kwargs is None:
            cost_kwargs = {}

        # setup evo strat workers
        sleep(5)  # wait on workers to connect
        num_workers = len(client.scheduler_info()['workers'])
        if num_workers == 0:
            raise ValueError("Error: there are no workers.")
        dask_workers = list(client.scheduler_info()['workers'].keys())
        if len(dask_workers) == 0:
            raise ValueError("Error: there are no workers.")
        evo_workers = [client.submit(initialize_worker, cost_kwargs,
                                     **self._kwargs, workers=[worker],
                                     pure=False)
                       for worker in dask_workers]

        # begin evolutionary steps
        for i in range(num_iterations):
            # update populations
            pop_update_jobs = []
            for worker_id in range(num_workers):
                pop_update_jobs.append(client.submit(
                    update_pops,
                    evo_workers[worker_id],
                    workers=[dask_workers[worker_id]],
                    pure=False))
            master_step = self._state.update_population()
            slave_steps = client.gather(pop_update_jobs)
            if any([master_step != slave_step for slave_step in slave_steps]):
                raise RuntimeWarning("Workers are out of sync on population"
                                     " update. Slave steps:", slave_steps,
                                     "Master step:", master_step)

            # submit jobs so all workers have work
            unworked_pops = list(range(self._state.population_size))
            jobs = []
            for index in range(num_workers):
                jobs.append(client.submit(
                    dispatch_work, cost_function,
                    evo_workers[index], unworked_pops.pop(), index,
                    workers=[dask_workers[index]],
                    pure=False))

            # submit work until whole population has been evaluated
            costs = [np.nan for _ in range(self._state.population_size)]
            working_batch = as_completed(jobs)
            for completed_job in working_batch:
                cost, pop_id, worker_id = completed_job.result()
                costs[pop_id] = cost
                if len(unworked_pops) > 0:
                    working_batch.add(
                        client.submit(dispatch_work,
                                      cost_function,
                                      evo_workers[worker_id],
                                      unworked_pops.pop(),
                                      worker_id,
                                      workers=[dask_workers[worker_id]],
                                      pure=False))
            if len(unworked_pops) > 0:
                raise RuntimeError("Failed to work all pops, remaining:", unworked_pops)
            if np.nan in costs:
                raise RuntimeError("Failed to assign cost for all pops:", costs)

            # update centroids
            centroid_update_jobs = []
            for worker_id in range(num_workers):
                centroid_update_jobs.append(client.submit(
                    scatter_cost,
                    costs,
                    evo_workers[worker_id],
                    workers=[dask_workers[worker_id]],
                    pure=False))

            # wait until centroids are updated
            master_step = self._state.update_centroid(costs)
            slave_steps = client.gather(centroid_update_jobs)
            if any([master_step != slave_step for slave_step in slave_steps]):
                raise RuntimeWarning("Workers are out of sync on centroid update."
                                     " Slave steps:", slave_steps,
                                     "Master step:", master_step)

            # update history
            self.generation_history.append(i)
            self.centroid_history.append(self._state.centroid.copy())
            self.sigma_history.append(self._state.sigma)
            self.cost_history.append(copy.copy(costs))
            if enable_path_history:
                self.sigma_path_history.append(np.mean(self._state._s_path))
                self.cov_path_history.append(np.mean(self._state._c_mat_path))

    @property
    def centroid(self):
        return self._state.centroid
