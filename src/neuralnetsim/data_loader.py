__all__ = ['load_as_matrix']


import numpy as np
from scipy.io import loadmat
from pathlib import Path


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

