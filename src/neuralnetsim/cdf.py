__all__ = ["ecdf", "eccdf"]


import numpy as np
from typing import Tuple


def ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the empirical CDF of a given set of data.
    :param data: A 1D array of values.
    :return: Tuple of (sorted data, cdf)
    """
    data = np.sort(data)
    size = float(len(data))
    all_unique = not (any(data[:-1] == data[1:]))
    if all_unique:
        cdf = np.array([i / size for i in range(0, len(data))])
    else:
        cdf = np.searchsorted(data, data, side='left') / size
        unique_data, unique_indices = np.unique(data, return_index=True)
        data = unique_data
        cdf = cdf[unique_indices]

    return data, cdf


def eccdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the complementary CDF.
    :param data: A 1D array of values.
    :return: Tuple of (sorted data, CCDF)
    """
    sorted_data, cdf = ecdf(data)
    return sorted_data, 1. - cdf
