import unittest

from llm4dfm.pipeline.preprocess import preprocess
from llm4dfm.pipeline.utils import extract_ex_num
from tests.preprocess.preprocess_utils import load_preprocess_datasets, load_exercise, get_info_from_filename, store_test_output


class PreprocessTest(unittest.TestCase):

    def test_preprocess(self):
        files = load_preprocess_datasets()

        output_generated = dict()
        output_expected = dict()

        for file_name in files:

            ex_output = load_exercise(file_name)

            exercise, version, prompt = get_info_from_filename(file_name)
            ex_num = extract_ex_num(exercise)
            is_demand = version == 'demand'

            output_generated[file_name] = []

            for out in ex_output['output']:
                step_preprocessed = dict()
                step_preprocessed['dependencies'], step_preprocessed['measures'], step_preprocessed['fact'] = preprocess(ex_num, out['dependencies'],
                       out['measures'] if out['measures'] else list(), out['fact'], is_demand)
                output_generated[file_name].append(step_preprocessed)

            output_expected[file_name] = ex_output['expected']

        for file in output_generated.keys():
            store_test_output(output_generated[file], file)



        for idx, file in enumerate(output_expected.keys()):
            self.assertEqual(output_expected[file], output_generated[file])  # add assertion here


if __name__ == '__main__':
    unittest.main()
