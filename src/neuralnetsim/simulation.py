__all__ = ["simulate_model"]


import neuralnetsim
import networkx as nx
import numpy as np
from distributed import Client
from pathlib import Path
from typing import Type
from typing import Dict
from typing import Any
from typing import Union


def simulation_worker(
        x0,
        parameter_path: Path,
        circuit_type: Union[Type[neuralnetsim.DistributionCircuit],
                            Type[neuralnetsim.NeuralCircuit]],
        graph: nx.DiGraph,
        rng: np.random.RandomState,
        duration,
        kernel_parameters
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
        duration: int,
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
        [
            {
                'x0': x0,
                'parameter_path': parameter_path,
                'circuit_type': circuit_type,
                'graph': graph,
                'rng': np.random.RandomState(rng.randint(1, 2**31)),
                'duration': duration,
                'kernel_parameters': kernel_parameters
            }

            for graph in range(fitted_graph_results['graphs'])
        ],
        pure=False
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
