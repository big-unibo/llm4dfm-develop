import unittest

from utils import load_files, get_info_from_filename, get_output_from_filename
from src.main.python.pipeline.metrics import MetricsCalculator as MetricsCalc

class MyTestCase(unittest.TestCase):

    def test_metrics(self):
        files = load_files()
        # Get ex_num to load gt

        for file in files:
            exercise_num, version, prompt = get_info_from_filename(file)
            print(exercise_num, version, prompt)

            ex_output = get_output_from_filename(file)

            print(ex_output)

            # Calculate metrics

            # if 'gt_preprocessed' in ex_output:
            #     ground_truth = ex_output['gt_preprocessed']
            # else:
            #     print('using standard ground truth')
            #     ground_truth = load_ground_truth_exercise(ex_config['gt'])
            #     if ex_config['demand']:
            #         ground_truth = ground_truth['demand_driven']
            #     else:
            #         ground_truth = ground_truth['supply_driven']
            #
            # metrics = []
            #
            # dep_gt = ground_truth['dependencies']
            # meas_gt = ground_truth['measures'] if ground_truth['measures'] else set()
            # fact_gt = ground_truth['fact']
            #
            # # Extracting ex number as last digit in exercise name
            # ex_num = extract_ex_num(ex_config['name'])
            #
            # metric_calc = MetricsCalc(fact_gt, meas_gt, dep_gt, ex_num, ex_config['demand'])




        self.assertEqual(True, True)  # add assertion here

if __name__ == '__main__':
    unittest.main()
