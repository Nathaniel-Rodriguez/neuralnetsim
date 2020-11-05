__all__ = ['load_as_matrix', 'load_spike_times',
           'save', 'load']


import numpy as np
import pickle
from scipy.io import loadmat
from pathlib import Path
from typing import List
from typing import Any


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


def save(data: Any, filename: Path, protocol=pickle.DEFAULT_PROTOCOL):
    """
    :param data: Save picklable Python object to file.
    :param filename: Save file path.
    :param protocol: Pickle protocol (default: python default).
    :return: None
    """
    pickled_obj_file = open(filename, 'wb')
    pickle.dump(data, pickled_obj_file, protocol=protocol)
    pickled_obj_file.close()


def load(filename: Path) -> Any:
    """
    Loads a pickled python object from file.
    :param filename: Save file path.
    :return: The loaded object.
    """
    pickled_obj_file = open(filename, 'rb')
    obj = pickle.load(pickled_obj_file)
    pickled_obj_file.close()
    return obj
