__all__ = ["DataManager",
           "TrainingManager"]


import numpy as np
from typing import Dict


class DataManager:
    """
    The DataManager splits neural spike-time data into training, validation, and
    test sets for a given number of folds. This class uses nested
    cross-validation to generate cross-validation splits. Nested cross-validation
    ensures that the model is trained only on data older than the validation
    and testing data. In turn, validation data is always older than test data.
    """
    def __init__(self, data: Dict[int, np.ndarray],
                 num_folds: int = 1,
                 test_ratio: float = 0.1,
                 start_buffer: float = 0.0):
        """
        :param data: A dictionary keyed by neuron id and valued by a 1-D numpy
            array of spike-times.
        :param num_folds: The number of desired folds for model selection.
        :param test_ratio: The fraction of data to set aside for testing.
            The validation ratio will be the same. All other data is left for
            training.
        :param start_buffer: Pad spike times by this amount at the start.
        """
        if num_folds < 1:
            raise ValueError("Number of folds {0} has to be greater than or"
                             " equal to 1.".format(str(num_folds)))

        if (num_folds * test_ratio + test_ratio) > 1.0:
            raise ValueError("Number of folds {0} not compatible"
                             " with test ratio {1}".format(str(num_folds),
                                                           str(test_ratio)))

        self._data = data
        self._num_folds = num_folds
        self._test_ratio = test_ratio
        self._start_buffer = start_buffer
        self._min_time = min(min(t for t in spike_times)
                             for spike_times in self._data.values()
                             if len(spike_times) > 0)
        self._max_time = max(max(t for t in spike_times)
                             for spike_times in self._data.values()
                             if len(spike_times) > 0)
        self._cap = np.arange(self._max_time, 0.0,
                              -self._test_ratio
                              * self._max_time)[:self._num_folds]
        self._test_bounds = np.arange(self._max_time, 0.0,
                                      -self._test_ratio
                                      * self._max_time)[1:self._num_folds + 1]
        self._validation_bounds = np.arange(self._max_time, 0.0,
                                            -self._test_ratio
                                            * self._max_time)[2:self._num_folds + 2]

    @property
    def data(self) -> Dict[int, np.ndarray]:
        """
        :return: The original data.
        """
        return self._data

    @data.setter
    def data(self, v):
        raise NotImplementedError

    def get_training_fold(self, fold: int) -> Dict[int, np.ndarray]:
        """
        Retrieves the training data for a given nested cross-validation fold.

        :param fold: The fold to draw the data from (0, num_folds-1).
        :return: A dictionary keyed by neuron id and valued by the spike times
            of that neuron.
        """
        if fold > (self._num_folds - 1):
            raise ValueError("Invalid fold {0}. Folds range"
                             " from 0 to num_folds-1".format(str(fold)))
        return {neuron: times[times < self._validation_bounds[fold]]
                        + self._start_buffer
                for neuron, times in self._data.items()}

    def get_validation_fold(self, fold: int) -> Dict[int, np.ndarray]:
        """
        Retrieves the validation data for a given nested cross-validation fold.

        :param fold: The fold to draw the data from (0, num_folds-1).
        :return: A dictionary keyed by neuron id and valued by the spike times
            of that neuron. The spike time origin is shifted by the start-time of
            the validation data.
        """
        if fold > (self._num_folds - 1):
            raise ValueError("Invalid fold {0}. Folds range"
                             " from 0 to num_folds-1".format(str(fold)))
        return {neuron: times[np.logical_and(times >= self._validation_bounds[fold],
                                             times < self._test_bounds[fold])]
                        - self._validation_bounds[fold] + self._start_buffer
                for neuron, times in self._data.items()}

    def get_test_fold(self, fold: int) -> Dict[int, np.ndarray]:
        """
        Retrieves the test data for a given nested cross-validation fold.

        :param fold: The fold to draw the data from (0, num_folds-1).
        :return: A dictionary keyed by neuron id and valued by the spike times
            of that neuron. The spike time origin is shifted by the start-time of
            the test data.
        """
        if fold > (self._num_folds - 1):
            raise ValueError("Invalid fold {0}. Folds range"
                             " from 0 to num_folds-1".format(str(fold)))
        return {neuron: times[np.logical_and(times >= self._test_bounds[fold],
                                             times <= self._cap[fold])]
                        - self._test_bounds[fold] + self._start_buffer
                for neuron, times in self._data.items()}

    def get_duration(self, split_type: str, fold: int, buffer: float = 0.0) -> float:
        """
        Returns a suggested run duration for a given split of the data. This
        will be the maximum spike time for that split + a buffer.

        :param split_type: Choose: "training", "validation", or "test"
        :param fold: Which fold to get the duration for.
        :param buffer: A buffer past the last spike (in ms) [default: 0.0].
        :return: A suggested run duration for a simulation.
        """
        if split_type == "training":
            return round(self._validation_bounds[fold] + buffer, 1)
        elif split_type == "validation":
            return round(self._test_bounds[fold] + buffer - self._validation_bounds[fold], 1)
        elif split_type == "test":
            return round(self._cap[fold] + buffer - self._test_bounds[fold], 1)
        else:
            raise ValueError("Invalid split_type of {0}. Must choose 'training',"
                             " 'validation', or 'test'.".format(split_type))


class TrainingManager:
    """
    Holds an internal state that can be used to carry out smaller batches of
    training steps from training data.
    """
    def __init__(self, data: Dict[int, np.ndarray],
                 duration: float,
                 batch_size: float,
                 start_buffer: float = 0.1):
        """
        :param data: Data from the data manager. Keyed by neuron id and
            valued by the spike times.
        :param duration: Full duration of data set as specified by DataManager.
        :param batch_size: Duration of a training batch in ms.
        :param start_buffer: Pad spike times by this amount at the start.
        """
        self._data = data
        self._duration = duration
        self._batch_size = batch_size
        self._start_buffer = start_buffer
        self._epoch = 0

    def get_duration(self):
        return self._batch_size

    def get_training_data(self) -> Dict[int, np.ndarray]:
        """
        :return: A subset of the training data for a given epoch. Epochs are
            incremented each time this method is called.
        """
        # if the end of the data set is reached, start at the beginning
        if (self._epoch + 1) * self._batch_size > self._duration:
            self._epoch = 0

        start = self._epoch * self._batch_size
        self._epoch += 1
        end = self._epoch * self._batch_size
        return {neuron: spikes[np.logical_and(spikes >= start,
                                              spikes < end)]
                + self._start_buffer
                for neuron, spikes in self._data.items()}
