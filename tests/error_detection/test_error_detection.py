import unittest
import traceback

from llm4dfm.pipeline.utils import load_ground_truth_exercise, extract_ex_num
from llm4dfm.pipeline.metrics import ErrorDetector
from llm4dfm.pipeline.preprocess import preprocess
from tests.error_detection.detection_utils import load_detection_datasets, load_detection_exercise, store_detection_test_output
from tests.utils import get_info_from_filename

class ErrorDetectionTest(unittest.TestCase):

    def test_error_detection(self):
        files = load_detection_datasets()
        # Get ex_num to load gt
        datection_gt = dict()
        detection_calculated = dict()

        output_generated = dict()

        for file_name in files:
            ex_output = load_detection_exercise(file_name)

            datection_gt[file_name] = ex_output['errors']

            if 'gt_preprocessed' in ex_output:
                ground_truth = ex_output['gt_preprocessed']
            elif 'ground_truth' in ex_output:
                    ground_truth = ex_output['ground_truth']
            else:
                # print('Calculating ground truth preprocess')
                exercise, version, prompt = get_info_from_filename(file_name)
                ex_num = extract_ex_num(exercise)

                is_demand = version == 'demand'
                ground_truth = load_ground_truth_exercise(exercise)
                if is_demand:
                    ground_truth = ground_truth['demand_driven']
                else:
                    ground_truth = ground_truth['supply_driven']


                ground_truth['dependencies'], ground_truth['measures'], ground_truth['fact'] = (
                    preprocess(ex_num, ground_truth['dependencies'], ground_truth['measures'] if ground_truth['measures'] else
                    list(), ground_truth['fact'], is_demand, list()))

            dep_gt = ground_truth['dependencies']
            meas_gt = ground_truth['measures'] if ground_truth['measures'] else list()
            fact_gt = ground_truth['fact']

            detector = ErrorDetector(fact_gt, meas_gt, dep_gt)

            detection_list = list()

            if 'output_preprocessed' not in ex_output:
                print(f'[DEBUG] Missing output preprocess for {file_name}, skipping')
                continue

            for i, output in enumerate(ex_output['output_preprocessed']):
                try:
                    detected = dict()
                    dep_output, meas_output, fact_output = (output['dependencies'],
                                                            output['measures'] if output['measures'] else list(),
                                                            output['fact'])
                    if 'metrics' in ex_output:
                        metrics = ex_output['metrics']
                        (detected['dependencies'], detected['measures'], detected['fact'], detected['attributes'],
                         detected['miscellaneous']) = detector.detect_with_metrics(fact_output, meas_output, dep_output,
                                                                                   metrics[i]['edges']['fn'],
                                                                                   metrics[i]['edges']['fp'])
                    else:
                        (detected['dependencies'], detected['measures'], detected['fact'], detected['attributes'],
                         detected['miscellaneous']) = detector.detect(fact_output, meas_output, dep_output)
                    detection_list.append(detected)
                except:
                    traceback.print_exc()
                    detection_list.append(dict())
                    print(f"Output {i}-th not correctly generated, skipped")

            detection_calculated[file_name] = detection_list
            output_generated[file_name] = {'errors': detection_list}

        for file in output_generated.keys():
            store_detection_test_output(output_generated[file], file)

        for file in datection_gt.keys():
            for idx, (detection_dict_gt, detection_dict_calc) in enumerate(zip(datection_gt[file], detection_calculated[file])):
                for comp in detection_dict_gt:
                    if type(comp) is dict:
                        for prop in comp:
                            try:
                                self.assertEqual(detection_dict_gt[comp][prop], detection_dict_calc[comp][prop])
                            except:
                                print(f'\n[DEBUG] Error in file *{file}* {idx}-th output: {comp}_{prop} Gt={detection_dict_gt[comp][prop]}, Out={detection_dict_calc[comp][prop]}')
                                raise AssertionError
                    else:
                        try:
                            self.assertEqual(detection_dict_gt[comp], detection_dict_calc[comp])
                        except:
                            print(
                                f'\n[DEBUG] Error in file *{file}* {idx}-th output: {comp} Gt={detection_dict_gt[comp]}, Out={detection_dict_calc[comp]}')
                            raise AssertionError

if __name__ == '__main__':
    unittest.main()
