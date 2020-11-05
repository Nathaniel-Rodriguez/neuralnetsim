__all__ = ["SCMAEvoStrat"]


import math
import numpy as np
from pathlib import Path
from typing import Dict
from typing import Callable
from typing import List
from time import sleep
from distributed import Client
from distributed import as_completed
from neuralnetsim.sliceops import *


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
        :param c_c: Covariance matrix time constant [defaulted].
        :param c_s: Sigma time constant [defaulted].
        :param c_1: First-rank covariance time constant [defaulted].
        :param c_mu: Mu-rank covariance time constant [defaulted].
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
        self._ordered_parents = np.zeros((self._mu, self._ndim))
        self._ordered_parents_buff = np.zeros((self._mu, self._ndim))
        self._s_path = np.zeros(self._ndim, dtype=dtype)
        self._weights = math.log(self._mu + 0.5)\
                        - np.log(np.arange(1, self._mu + 1))
        self._weights /= sum(self._weights)
        self._weights.astype(dtype, copy=False)
        self._mu_eff = 1. / sum(self._weights ** 2) if mu_eff is None else mu_eff
        self._c_c = 4. / (self._ndim + 4) if c_c is None else c_c
        self._c_s = (self._mu_eff + 2) / (self._ndim + self._mu_eff + 3) \
            if c_s is None else c_s
        self._c_1 = 2. / ((self._ndim + 1.3)**2 + self._mu_eff)\
                    * (self._ndim + 2) / 3 \
            if c_1 is None else c_1
        self._c_mu = min([1 - self._c_1, 2.
                          * (self._mu_eff - 2 + 1 / self._mu_eff)
                          / ((self._ndim + 2)**2 + self._mu_eff)]) \
            if c_mu is None else c_mu
        self._d_s = 1 + 2 * max(0., math.sqrt((self._mu_eff - 1)
                                              / (self._ndim + 1)) - 1) + self._c_s \
            if d_s is None else d_s

        self._d_diag = self._c_mat**0.5

    def update_population(self):
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
        np.sum(self._population, self._mutations, out=self._population)

    def update_centroid(self, costs: List[float]):
        # sort costs and then add weighted parents to centroid
        sorted_parent_indices = np.argsort(costs)
        self._x_last[:] = self._x[:]
        for i, parent_index in enumerate(sorted_parent_indices[:self._mu]):
            self._ordered_parents[i, :] = self._population[parent_index, :]
            self._x_buffer[:] = self._population[parent_index, :]
            np.multiply(self._x_buffer, self._weights[i], out=self._x_buffer)
            self._x += self._x_buffer

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
        np.multiply(self._weights, self._ordered_parents.T, out=self._ordered_parents_buff.T)
        np.multiply(self._ordered_parents_buff.T, self._ordered_parents.T, out=self._ordered_parents_buff.T)
        np.sum(self._ordered_parents_buff.T, axis=-1, out=self._ndim_buff2)
        np.multiply(self._ndim_buff2, self._c_mu, out=self._ndim_buff2)
        np.add(self._c_mat, self._ndim_buff, out=self._c_mat)
        np.add(self._c_mat, self._ndim_buff2, out=self._c_mat)

        # update sigma
        self._s *= np.exp((np.linalg.norm(self._s_path)
                           / self._s_dim - 1.) * self._c_s / self._d_s)

        # update diagonal
        np.sqrt(self._c_mat, out=self._d_diag)


class EvoStratWorker:
    def __init__(self, *args, **kwargs):
        self._state = EvoStratState()

    def retrieve_pop(self, pop_id: int):
        return self._state.centroid[pop_id]

    def update(self, fitnesses: List[float]):
        self._state.update_centroid(fitnesses)


def initialize_worker(*args, **kwargs):
    print("INITIALIZED", flush=True)
    return EvoStratWorker(*args, **kwargs)


def scatter_fitness(fitness_array, worker):
    print("SCATTER", flush=True)
    worker.update(fitness_array)
    sleep(1)
    return None


def dispatch_work(fitness_function: Callable[[np.ndarray], float],
                  worker: EvoStratWorker,
                  pop_id: int,
                  worker_id: int,
                  fitness_kwargs: Dict):
    print("WORK DISPATCH", pop_id, worker_id, flush=True)
    return fitness_function(worker.retrieve_pop(pop_id),
                            **fitness_kwargs), pop_id, worker_id


class SCMAEvoStrat:
    def __init__(self):
        self._state = EvoStratState()

    def to_file(self):
        pass  # just save some info to the pyboj, not the whole class

    def run(self, fitness_function: Callable[[np.ndarray], float],
            client: Client, num_iterations, fitness_kwargs: Dict = None):
        if fitness_kwargs is None:
            fitness_kwargs = {}
        # setup evo strat workers
        sleep(5)  # wait on workers to connect
        print("DID SLEEP")
        num_workers = len(client.scheduler_info()['workers'])
        if num_workers == 0:
            raise ValueError("Error: there are no workers.")
        dask_workers = list(client.scheduler_info()['workers'].keys())
        if len(dask_workers) == 0:
            raise ValueError("Error: there are no workers.")
        print("GOT WORKERS... initializing")
        evo_workers = [client.submit(initialize_worker, {}, workers=[worker])
                       for worker in dask_workers]
        print("ALL INITIALIZED")
        # draw from centroids

        for i in range(num_iterations):
            print("LOOP", i)
            # submit jobs so all workers have work
            unworked_pops = list(range(self._state.population_size))
            jobs = []
            for index in range(num_workers):
                print("SUBMIT JOB")
                jobs.append(client.submit(
                    dispatch_work, fitness_function,
                    evo_workers[index], unworked_pops.pop(), index,
                    fitness_kwargs,
                    workers=[dask_workers[index]]))

            # submit work until whole population has been evaluated
            worked_pops = []
            working_batch = as_completed(jobs)
            for completed_job in working_batch:
                fitness, pop_id, worker_id = completed_job.result()
                print("JOB COMPLETED", fitness, pop_id, worker_id)
                worked_pops.append((fitness, pop_id))

                if len(unworked_pops) > 0:
                    print("MAKE NEW WORK")
                    working_batch.add(
                        client.submit(dispatch_work,
                                      fitness_function,
                                      evo_workers[worker_id],
                                      unworked_pops.pop(),
                                      worker_id,
                                      fitness_kwargs,
                                      workers=[dask_workers[worker_id]]))

            # update centroids
            centroid_update_jobs = []
            for worker_id in range(num_workers):
                print("DO CENTROIDS")
                centroid_update_jobs.append(client.submit(
                    scatter_fitness,
                    [],
                    evo_workers[worker_id],
                    workers=[dask_workers[worker_id]]))

            # wait until centroids are updated
            client.gather(centroid_update_jobs)
