import unittest
import neuralnetsim
from pkg_resources import resource_exists
from pkg_resources import resource_filename


class TestDataLoader(unittest.TestCase):
    def test_load_matrix(self):
        self.assertTrue(resource_exists("tests", "test_data/pdf.mat"))
        file_path = resource_filename("tests", "test_data/pdf.mat")
        mat = neuralnetsim.load_as_matrix(file_path, "pdf")
        self.assertEqual(mat.shape[0], 243)
        self.assertEqual(mat.shape[1], 243)

    def test_load_spikes(self):
        file_path = resource_filename("tests", "test_data/data.mat")
        data = neuralnetsim.load_spike_times(file_path)
        self.assertEqual(len(data), 243)
        self.assertEqual(data[0].shape[0], 6528)


if __name__ == '__main__':
    unittest.main()
