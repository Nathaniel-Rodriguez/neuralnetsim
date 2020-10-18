import unittest
import neuralnetsim


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
    pass
