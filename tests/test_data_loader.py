import neuralnetsim
from pkg_resources import resource_filename
import unittest


class TestDataLoader(unittest.TestCase):
    def test_load_matrix(self):
        file_path = resource_filename("neuralnetsim", "tests/test_data/pdf.mat")
        mat = neuralnetsim.load_as_matrix(file_path, "pdf")
        self.assertEqual(mat.shape[0], 384)
        self.assertEqual(mat.shape[1], 384)


if __name__ == '__main__':
    unittest.main()