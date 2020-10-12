__all__ = ['load_as_matrix']


import numpy as np
from scipy.io import loadmat
from pathlib import Path


def load_as_matrix(file_path: Path, key: str) -> np.ndarray:
    """
    Opens and loads a matrix from a matlab file when given the data key.
    :param file_path: Path to a file containing a matrix.
    :param key: Key for the matrix.
    :return: A 2D numpy array with the matrix elements.
    """
    return loadmat(str(file_path))[key]

