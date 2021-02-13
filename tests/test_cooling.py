import unittest
import neuralnetsim
import numpy as np


class TestExponentialSchedule(unittest.TestCase):
    def test_initial_t(self):
        cooler = neuralnetsim.ExponentialCoolingSchedule(1.0, 1.0, 0)
        self.assertAlmostEqual(1.0, cooler.step())

    def test_step(self):
        cooler = neuralnetsim.ExponentialCoolingSchedule(1.0, 0.9, 0)
        cooler.step()
        self.assertAlmostEqual(cooler.step(), 0.90)
        self.assertAlmostEqual(cooler.step(), 0.81)

    def test_test_final_t(self):
        cooler = neuralnetsim.ExponentialCoolingSchedule(1.0, 0.9, 0)
        for i in range(1000):
            cooler.step()
        self.assertAlmostEqual(cooler.t, 0.0)

    def test_start(self):
        cooler = neuralnetsim.ExponentialCoolingSchedule(1.0, 0.9, 50)
        for i in range(51):
            cooler.step()
        self.assertAlmostEqual(cooler.t, 1.0)
        self.assertAlmostEqual(cooler.step(), 0.90)
        self.assertAlmostEqual(cooler.step(), 0.81)

    def test_stop(self):
        cooler = neuralnetsim.ExponentialCoolingSchedule(1.0, 0.9, 0, 50)
        for i in range(50):
            cooler.step()
        test_t = cooler.t
        for i in range(50):
            self.assertAlmostEqual(cooler.step(), test_t)


class TestAdaptiveSchedule(unittest.TestCase):
    def test_initial_t(self):
        cooler = neuralnetsim.AdaptiveCoolingSchedule(1.0, 2.0, 10, 0.1, 5,
                                                      0, 50, 0.0)
        self.assertAlmostEqual(1.0, cooler.step(0.0))

    def test_step(self):
        cooler = neuralnetsim.AdaptiveCoolingSchedule(1.0, 2.0, 2, 0.1, 2,
                                                      0, 50, 0.0)
        self.assertAlmostEqual(cooler.step(0.0), 1.0)
        self.assertAlmostEqual(cooler.step(0.0), 1.0)
        self.assertAlmostEqual(cooler.thistory[0], 1.0)
        self.assertAlmostEqual(cooler.thistory[1], 1.0)
        self.assertAlmostEqual(cooler.ehistory[0], 0.0)
        self.assertAlmostEqual(cooler.ehistory[1], 0.0)
        self.assertAlmostEqual(cooler.thistory[0], 1.0)
        cooler.step(1.0)
        self.assertAlmostEqual(cooler.ehistory[0], 0.0)
        self.assertAlmostEqual(cooler.ehistory[1], 1.0)
        t = cooler.step(1.0)
        self.assertAlmostEqual(cooler.thistory[1], t)

    def test_test_final_t(self):
        cooler = neuralnetsim.AdaptiveCoolingSchedule(1.0, 2.0, 10, 1.0, 10,
                                                      tmin=0.01)
        rng = np.random.RandomState(15135)
        energies = rng.normal(size=100)
        for e in energies:
            self.assertGreaterEqual(cooler.step(e), 0.01)

    def test_stop(self):
        cooler = neuralnetsim.AdaptiveCoolingSchedule(1.0, 2.0, 10, 1.0, 10, 0,
                                                      stop=25)
        rng = np.random.RandomState(15135)
        energies = rng.normal(size=100)
        for i in range(25):
            cooler.step(energies[i])
        test_t = cooler.t
        for i in range(75):
            self.assertAlmostEqual(cooler.step(energies[i]), test_t)
