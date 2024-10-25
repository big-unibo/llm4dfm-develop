import unittest
from tests.preprocess.utils import load_preprocess_datasets

class PreprocessTest(unittest.TestCase):

    def test_preprocess(self):
        files = load_preprocess_datasets()
        print(files)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
