__all__ = ["simulate_model",
           "simulate_grid",
           "simulate_orig"]


import neuralnetsim
import networkx as nx
import numpy as np
from distributed import Client
from pathlib import Path
from typing import Type
from typing import Dict
from typing import Any
from typing import List
from typing import Union


def simulation_worker(
        graph: nx.DiGraph,
        rng: np.random.RandomState,
        x0: np.ndarray,
        parameter_path: Path,
        circuit_type: Union[Type[neuralnetsim.DistributionCircuit],
                            Type[neuralnetsim.NeuralCircuit]],
        duration: float,
        kernel_parameters: Dict
) -> Dict[int, np.ndarray]:
    """


    :param x0:
    :param parameter_path:
    :param circuit_type:
    :param graph:
    :param rng:
    :param duration:
    :param kernel_parameters:
    :return:
    """
    circuit_parameters = neuralnetsim.load(parameter_path)
    circuit_parameters.network = graph
    circuit_parameters.from_optimizer(x0)
    with neuralnetsim.CircuitManager(circuit_type, kernel_parameters,
                                     circuit_parameters, rng) as circuit:
        circuit.run(duration)
        return circuit.get_spike_trains()


def simulate_model(
        x0,
        parameter_path: Path,
        fitted_graph_path: Path,
        name: str,
        client: Client,
        duration: float,
        seed: int,
        circuit_type: Type,
        save_path: Path,
        kernel_parameters: Dict[str, Any] = None,
):
    """
    Data in list is matched to the order of the graphs in the fitted graph
    result file.

    :param x0:
    :param parameter_path:
    :param fitted_graph_path:
    :param name:
    :param client:
    :param duration:
    :param seed:
    :param circuit_type:
    :param save_path:
    :param kernel_parameters:
    :return:
    """
    if kernel_parameters is None:
        kernel_parameters = {}
    fitted_graph_results = neuralnetsim.load(fitted_graph_path)
    rng = np.random.RandomState(seed)
    sims = client.map(
        simulation_worker,
        [graph for graph in fitted_graph_results['graphs']],
        [np.random.RandomState(rng.randint(1, 2**31))
         for _ in range(len(fitted_graph_results['graphs']))],
        pure=False,
        x0=x0,
        parameter_path=parameter_path,
        circuit_type=circuit_type,
        duration=duration,
        kernel_parameters=kernel_parameters
    )
    data = client.gather(sims)
    neuralnetsim.save(
        {
            'spike_data': data,
            'seed': seed,
            'name': name,
            'duration': duration,
            'kernel_parameters': kernel_parameters
        },
        save_path
    )


def grid_worker(
        graph: nx.DiGraph,
        rng: np.random.RandomState,
        par: float,
        x0: np.ndarray,
        par_key: str,
        parameter_path: Path,
        circuit_type: Union[Type[neuralnetsim.DistributionCircuit],
                            Type[neuralnetsim.NeuralCircuit]],
        duration: float,
        kernel_parameters: Dict
) -> Dict[int, np.ndarray]:
    """

    :param graph:
    :param rng:
    :param par:
    :param x0:
    :param par_key:
    :param parameter_path:
    :param circuit_type:
    :param duration:
    :param kernel_parameters:
    :return:
    """
    kernel_parameters.update({'grng_seed': rng.randint(1, 2e5),
                              **kernel_parameters})
    circuit_parameters = neuralnetsim.load(parameter_path)
    circuit_parameters.network = graph
    circuit_parameters.from_optimizer(x0)
    circuit_parameters.set_par(par_key, par)
    with neuralnetsim.CircuitManager(circuit_type, kernel_parameters,
                                     circuit_parameters, rng) as circuit:
        circuit.run(duration)
        return circuit.get_spike_trains()


def simulate_grid(
        x0,
        par_range: Union[List[float], np.ndarray],
        par_key: str,
        parameter_path: Path,
        fitted_graph_path: Path,
        name: str,
        client: Client,
        duration: float,
        seed: int,
        circuit_type: Type,
        save_path: Path,
        kernel_parameters: Dict[str, Any] = None,
):
    """

    :param x0:
    :param par_range:
    :param par_key:
    :param parameter_path:
    :param fitted_graph_path:
    :param name:
    :param client:
    :param duration:
    :param seed:
    :param circuit_type:
    :param save_path:
    :param kernel_parameters:
    :return:
    """
    if kernel_parameters is None:
        kernel_parameters = {}
    fitted_graph_results = neuralnetsim.load(fitted_graph_path)
    rng = np.random.RandomState(seed)
    num_graphs = range(len(fitted_graph_results['graphs']))
    sims = client.map(
        grid_worker,
        [graph
         for _ in par_range
         for graph in fitted_graph_results['graphs']],
        [np.random.RandomState(rng.randint(1, 2**31))
         for _ in par_range
         for _ in num_graphs],
        [par for par in par_range
         for _ in num_graphs],
        pure=False,
        x0=x0,
        par_key=par_key,
        parameter_path=parameter_path,
        circuit_type=circuit_type,
        duration=duration,
        kernel_parameters=kernel_parameters
    )
    data = client.gather(sims)
    neuralnetsim.save(
        {
            'spike_data': data,
            'original_graph': fitted_graph_results['original'],
            'graphs': [graph for _ in par_range
                       for graph in fitted_graph_results['graphs']],
            'target_modularities':
                [mu for _ in par_range
                 for mu in fitted_graph_results['target_modularities']],
            'grid_par': [par for par in par_range for _ in num_graphs],
            'par_key': par_key,
            'seed': seed,
            'name': name,
            'duration': duration,
            'kernel_parameters': kernel_parameters
        },
        save_path
    )


def orig_worker(
        rng: np.random.RandomState,
        par: float,
        graph: nx.DiGraph,
        x0: np.ndarray,
        par_key: str,
        parameter_path: Path,
        circuit_type: Union[Type[neuralnetsim.DistributionCircuit],
                            Type[neuralnetsim.NeuralCircuit]],
        duration: float,
        kernel_parameters: Dict
):
    kernel_parameters.update({'grng_seed': rng.randint(1, 2e5),
                              **kernel_parameters})
    circuit_parameters = neuralnetsim.load(parameter_path)
    circuit_parameters.network = graph
    circuit_parameters.from_optimizer(x0)
    circuit_parameters.set_par(par_key, par)
    with neuralnetsim.CircuitManager(circuit_type, kernel_parameters,
                                     circuit_parameters, rng) as circuit:
        circuit.run(duration)
        return circuit.get_spike_trains()


def simulate_orig(
        x0,
        par_range: Union[List[float], np.ndarray],
        par_key: str,
        parameter_path: Path,
        orig_graph_path: Path,
        n_trials: int,
        client: Client,
        duration: float,
        seed: int,
        circuit_type: Type,
        save_path: Path,
        kernel_parameters: Dict[str, Any] = None,
):
    """

    :param x0:
    :param par_range:
    :param par_key:
    :param parameter_path:
    :param fitted_graph_path:
    :param name:
    :param client:
    :param duration:
    :param seed:
    :param circuit_type:
    :param save_path:
    :param kernel_parameters:
    :return:
    """
    if kernel_parameters is None:
        kernel_parameters = {}
    graph = neuralnetsim.load(orig_graph_path)
    rng = np.random.RandomState(seed)
    sims = client.map(
        orig_worker,
        [np.random.RandomState(rng.randint(1, 2**31))
         for _ in par_range
         for _ in range(n_trials)],
        [par for par in par_range
         for _ in range(n_trials)],
        pure=False,
        x0=x0,
        graph=graph,
        par_key=par_key,
        parameter_path=parameter_path,
        circuit_type=circuit_type,
        duration=duration,
        kernel_parameters=kernel_parameters
    )
    data = client.gather(sims)
    neuralnetsim.save(
        {
            'spike_data': data,
            'control_var': [par for par in par_range for _ in range(n_trials)],
            'control_key': par_key,
            'seed': seed,
            'duration': duration
        },
        save_path
    )
