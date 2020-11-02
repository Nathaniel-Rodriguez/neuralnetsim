__all__ = ['load_as_matrix', 'load_spike_times']


import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import List


def load_as_matrix(file_path: Path, key: str) -> np.ndarray:
    """
    Opens and loads a matrix from a matlab file when given the data key.
    Popular keys include 'data' for spike-trains, 'weights' for TE matrix,
    'pdf' for binary significance matrix, 'x' for x-positions, 'y' for
    y-positions.
    :param file_path: Path to a file containing a matrix.
    :param key: Key for the matrix.
    :return: A an array of the data.
    """
    mat = loadmat(str(file_path))
    try:
        return mat[key]
    except KeyError as err:
        print("Failed to find key:", key, "\n",
              "The following keys are available:", list(mat.keys()))
        raise err


def load_spike_times(file_path: Path) -> List[np.ndarray]:
    """
    Loads spike time data from a matlab file.
    :param file_path: Path to a file containing spike time data.
    :return: A list of 1-D numpy arrays with spike times.
    """
    raw_data = load_as_matrix(file_path, "data")
    return [raw[0] for raw in raw_data[0]]
