__all__ = ["fit_network"]


import networkx as nx
import random
import numpy as np
from neuralnetsim.cooling import AdaptiveCoolingSchedule
from neuralnetsim.energy import NeuralEnergyFunction
from neuralnetsim.annealing import NetworkAnnealer
from neuralnetsim import save
from pathlib import Path
from typing import List
from typing import Dict
from typing import Any
from typing import Union
from distributed import Client


def network_fitting_worker(kwargs: Dict) -> nx.DiGraph:
    """
    A Dask worker function for the fit_network function.
    :param kwargs: A dictionary of keyword arguments passed to the worker.
    :return: A fitted network.
    """
    energy = NeuralEnergyFunction(kwargs['graph'],
                                  kwargs['target_modularity'],
                                  **kwargs['energy_kwargs'])
    cooling = AdaptiveCoolingSchedule(**kwargs['cooling_kwargs'])
    annealer = NetworkAnnealer(cooling_schedule=cooling,
                               energy_function=energy,
                               seed=kwargs['seed'],
                               **kwargs['annealing_kwargs'])
    return annealer.fit_predict(kwargs['graph'])


def fit_network(graph: nx.DiGraph,
                client: Client,
                target_modularities: Union[List[float], np.ndarray],
                graphs_per_modularity: int,
                energy_kwargs: Dict[str, Any],
                cooling_kwargs: Dict[str, Any],
                annealing_kwargs: Dict[str, Any],
                seed: int,
                save_dir: Path = Path.cwd(),
                prefix: str = "test"):
    """
    Initiates a Dask run that fits a given number of graphs to multiple
    modularities. Saves the results in a `fit_newtwork_results.pyobj` file.
    This is a Python object file.
    :param graph: The graph that will be fit too.
    :param client: A Dask client for parallel distribution.
    :param target_modularities: A list of target modularities to fit networks too.
    :param graphs_per_modularity: The number of fitted graphs to generated for
    each target modularity.
    :param energy_kwargs: A dictionary of keyword arguments for the energy function.
    :param cooling_kwargs: A dictionary of keyword arguments for the cooling
    schedule.
    :param annealing_kwargs: A dictionary of keyword arguments for the annealer.
    :param seed: Used for generating seeds for the annealers.
    :param save_dir: The directory in which to save the results.
    :param prefix: A filename prefix to prepend to the output file.
    :return: None
    """
    rng = random.Random(seed)
    annealers = client.map(
        network_fitting_worker,
        [{'graph': graph, 'target_modularity': modularity,
          'trial': trial, 'energy_kwargs': energy_kwargs,
          'cooling_kwargs': cooling_kwargs,
          'annealing_kwargs': annealing_kwargs,
          'seed': rng.randint(0, 2**31)}
         for modularity in target_modularities
         for trial in range(graphs_per_modularity)],
        pure=False)
    fitted_graphs = client.gather(annealers)
    results = {
        'original': graph,
        'graphs': fitted_graphs,
        'trials': [trial for _ in target_modularities
                   for trial in range(graphs_per_modularity)],
        'target_modularities': [
            modularity for modularity in target_modularities
            for _ in range(graphs_per_modularity)
        ],
        'parameters': {'modularities': target_modularities,
                       'trials': graphs_per_modularity,
                       'energy_kwargs': energy_kwargs,
                       'cooling_kwargs': cooling_kwargs,
                       'annealing_kwargs': annealing_kwargs,
                       'seed': seed,
                       'prefix': prefix}
    }
    save(results, save_dir.joinpath(prefix + "_fit_network_results.pyobj"))
