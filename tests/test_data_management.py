import unittest
import neuralnetsim
import numpy as np


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.data = {0: np.array([0.0, 1.0, 4.0, 6.0, 7.5, 9.5, 10.0]),
                     2: np.array([1.0, 2.0, 5.0, 8.2, 9.0])}

    def test_properties(self):
        dm = neuralnetsim.DataManager(self.data, 1, test_ratio=0.1)
        self.assertAlmostEqual(dm._max_time, 10.0, 3)
        self.assertAlmostEqual(dm._test_bounds[0], 9.0, 3)
        self.assertAlmostEqual(dm._validation_bounds[0], 8.0, 3)

        dm = neuralnetsim.DataManager(self.data, 2, test_ratio=0.1)
        self.assertAlmostEqual(dm._test_bounds[1], 8.0, 3)
        self.assertAlmostEqual(dm._validation_bounds[1], 7.0, 3)

        self.assertRaises(ValueError, neuralnetsim.DataManager, self.data, 99, 0.1)
        self.assertRaises(ValueError, neuralnetsim.DataManager, self.data, -1, 0.1)

    def test_training_fold(self):
        dm = neuralnetsim.DataManager(self.data, 1, test_ratio=0.1)
        training = dm.get_training_fold(0)
        self.assertEqual(len(training[0]), 5)
        self.assertEqual(len(training[2]), 3)
        self.assertAlmostEqual(training[0][0], 0.0)
        self.assertAlmostEqual(training[0][1], 1.0)
        self.assertAlmostEqual(training[0][2], 4.0)
        self.assertAlmostEqual(training[0][3], 6.0)
        self.assertAlmostEqual(training[2][0], 1.0)
        self.assertAlmostEqual(training[2][1], 2.0)
        self.assertAlmostEqual(training[2][2], 5.0)

        dm = neuralnetsim.DataManager(self.data, 3, test_ratio=0.1)
        training = dm.get_training_fold(2)
        self.assertEqual(len(training[0]), 3)
        self.assertEqual(len(training[2]), 3)

    def test_validation_fold(self):
        dm = neuralnetsim.DataManager(self.data, 1, test_ratio=0.1)
        val = dm.get_validation_fold(0)
        self.assertEqual(len(val[0]), 0)
        self.assertEqual(len(val[2]), 1)
        self.assertAlmostEqual(val[2][0], 0.2)

        dm = neuralnetsim.DataManager(self.data, 2, test_ratio=0.1)
        val = dm.get_validation_fold(1)
        self.assertEqual(len(val[0]), 1)
        self.assertEqual(len(val[2]), 0)
        self.assertAlmostEqual(val[0][0], 0.5)

    def test_test_fold(self):
        dm = neuralnetsim.DataManager(self.data, 1, test_ratio=0.1)
        test = dm.get_test_fold(0)
        self.assertEqual(len(test[0]), 2)
        self.assertEqual(len(test[2]), 1)
        self.assertAlmostEqual(test[0][0], 0.5)
        self.assertAlmostEqual(test[2][0], 0.0)

        dm = neuralnetsim.DataManager(self.data, 2, test_ratio=0.1)
        test = dm.get_test_fold(1)
        self.assertEqual(len(test[0]), 0)
        self.assertEqual(len(test[2]), 2)
        self.assertAlmostEqual(test[2][0], 0.2)
        self.assertAlmostEqual(test[2][1], 1.0)
