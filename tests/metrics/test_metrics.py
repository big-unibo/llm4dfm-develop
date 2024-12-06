import unittest
import traceback

from llm4dfm.pipeline.utils import load_ground_truth_exercise, extract_ex_num, label_edges
from llm4dfm.pipeline.metrics import MetricsCalculator
from llm4dfm.pipeline.preprocess import preprocess
from tests.metrics.metrics_utils import load_metrics_datasets, load_metrics_exercise, store_metrics_test_output
from tests.utils import get_info_from_filename

class MetricsTest(unittest.TestCase):

    def test_metrics(self):
        files = load_metrics_datasets()
        # Get ex_num to load gt
        metrics_gt = dict()
        metrics_calculated = dict()

        output_generated = dict()

        for file_name in files:
            ex_output = load_metrics_exercise(file_name)

            metrics_gt[file_name] = ex_output['metrics']

            # Default configurations
            ex_num, is_demand = '', False

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


            metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt, ex_num, is_demand)

            metrics_list = []
            output_preprocessed = []

            for i, output in enumerate(ex_output['output']):
                try:
                    dep_output, meas_output, fact_output = preprocess(ex_num, output['dependencies'],
                                                                      output['measures'] if output['measures'] else list(),
                                                                      output['fact'], is_demand, ground_truth['dependencies'])
                    edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used = metric_calc.get_edges_idx(fact_output, meas_output,
                                                                                                  dep_output)

                    tp_nodes, fp_nodes, fn_nodes = metric_calc.get_nodes()

                    step_metric = {
                        'edges': metric_calc.calculate_metrics_from_preprocessed(edges_tp_idx, edges_fp_idx, edges_fn_idx),
                        'nodes': metric_calc.calculate_metrics_nodes(fact_output, meas_output, dep_output)}
                    metrics_calculated[file_name] = step_metric
                    metrics_list.append(step_metric)

                    output_to_use = {'dependencies': dep_output, 'measures': meas_output, 'fact': fact_output}

                    out, gt = label_edges(output_to_use, ground_truth, edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used)

                    output_preprocessed.append(
                        {'dependencies': out['dependencies'], 'fact': out['fact'], 'measures': out['measures'],
                         'ground_truth_labels': gt, 'nodes': {'tp': list(tp_nodes), 'fp': list(fp_nodes),
                                                              'fn': list(fn_nodes)}})

                except:
                    traceback.print_exc()
                    metrics_list.append(dict())
                    print(f"Output {i}-th not correctly generated, skipped")

            metrics_calculated[file_name] = metrics_list
            output_generated[file_name] = {'output_preprocessed': output_preprocessed, 'gt_preprocessed': ground_truth,
                                           'metrics': metrics_list}

        for file in output_generated.keys():
            store_metrics_test_output(output_generated[file], file)

        for file in metrics_gt.keys():
            for idx, (metr_dict_gt, metr_dict_calc) in enumerate(zip(metrics_gt[file], metrics_calculated[file])):
                for comp in metr_dict_gt:
                    for prop in ['tp', 'fp', 'fn']:
                        try:
                            self.assertEqual(metr_dict_gt[comp][prop], metr_dict_calc[comp][prop])
                        except:
                            print(f'\n[DEBUG] Error in file *{file}* {idx}-th output: {comp}_{prop} Gt={metr_dict_gt[comp][prop]}, Out={metr_dict_calc[comp][prop]}')
                            raise AssertionError

if __name__ == '__main__':
    unittest.main()
