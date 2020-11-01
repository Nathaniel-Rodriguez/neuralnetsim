import unittest
import neuralnetsim


class TestValueTranslator(unittest.TestCase):
    def test_to_model(self):
        translator = neuralnetsim.ValueTranslator("test", 0, 10)
        self.assertAlmostEqual(translator.to_model(1.0), 10.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_model(0.9), 9.0, 3)
        self.assertAlmostEqual(translator.to_model(0.1), 1.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -10, 10)
        self.assertAlmostEqual(translator.to_model(0.5), 0.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), -10.0, 3)
        self.assertAlmostEqual(translator.to_model(1.0), 10.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -20, -10)
        self.assertAlmostEqual(translator.to_model(0.5), -15.0, 3)
        self.assertAlmostEqual(translator.to_model(0.0), -20.0, 3)
        self.assertAlmostEqual(translator.to_model(1.0), -10.0, 3)
        self.assertAlmostEqual(translator.to_model(-0.5), -15.0, 3)
        self.assertAlmostEqual(translator.to_model(2.0), -20.0, 3)

    def test_to_optimizer(self):
        translator = neuralnetsim.ValueTranslator("test", 0, 10)
        self.assertAlmostEqual(translator.to_optimizer(10.0), 1.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(0.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(9.0), 0.9, 3)
        self.assertAlmostEqual(translator.to_optimizer(1.0), 0.1, 3)

        translator = neuralnetsim.ValueTranslator("test", -10, 10)
        self.assertAlmostEqual(translator.to_optimizer(0.0), 0.5, 3)
        self.assertAlmostEqual(translator.to_optimizer(-10.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(10.0), 1.0, 3)

        translator = neuralnetsim.ValueTranslator("test", -20, -10)
        self.assertAlmostEqual(translator.to_optimizer(-15.0), 0.5, 3)
        self.assertAlmostEqual(translator.to_optimizer(-20.0), 0.0, 3)
        self.assertAlmostEqual(translator.to_optimizer(-10.0), 1.0, 3)
