import argparse
import traceback
from copy import deepcopy

from utils import load_ground_truth_exercise, load_output_exercise, load_yaml_from_resources, \
    extract_ex_num, label_edges, store_additional_properties

from preprocess import preprocess

def _calc_metrics(tp, fn, fp):
    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1, tp_count, fn_count, fp_count

def _load_nodes(dependencies):
    return set(','.join(subset) for sublist in dependencies for subset in sublist[:2])

class MetricsCalculator:

    def __init__(self, gt_fact, gt_measures, gt_dependencies, ex_number, demand=False):
        self.gt_raw = dict()
        self.out_raw = dict()
        self.gt_preprocessed = dict()
        self.gt_edges_list = list()
        self.gt_nodes_set = set()
        self.out_preprocessed = dict()
        self.out_edges_list = list()
        self.out_nodes_set = set()
        self.ex_number = ex_number
        self._load_gt(gt_fact, gt_measures, gt_dependencies)
        self.demand = demand

    def _load_gt(self, gt_fact, gt_measures, gt_dependencies):
        self.gt_raw['fact'] = gt_fact
        self.gt_raw['measures'] = gt_measures
        self.gt_raw['dependencies'] = gt_dependencies

    def get_edges_idx(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        return self._calculate_edges_indexes()

    def _get_nodes(self, dep_gt, dep_generated, measure_gt, measure_generated, fact_gt, fact_generated):

        tp, fn, fp, gt_used = set(), set(), set(), set()

        dep_gt_to_iterate = deepcopy(dep_gt)

        for dep in dep_generated:
            inserted = False
            for dp_gt in dep_gt_to_iterate:
                if {dp.lower() for dp in dep.split(',')} == {dp.lower() for dp in dp_gt.split(',')}:
                    inserted = True
                    tp.add(dep)
                    gt_used.add(dp_gt)
                    break
            if not inserted:
                fp.add(dep)
            else:
                dep_gt_to_iterate.discard(dep)
        for dp_gt in dep_gt:
            if dp_gt not in gt_used:
                fn.add(dp_gt)

        meas_gt_to_iterate = deepcopy(measure_gt)

        tp_meas, fn_meas, fp_meas, gt_meas_used = set(), set(), set(), set()

        for meas in measure_generated:
            inserted = False
            for meas_gt in meas_gt_to_iterate:
                if meas.lower() == meas_gt.lower():
                    inserted = True
                    tp_meas.add(meas)
                    gt_meas_used.add(meas)
                    break
            if not inserted:
                fp_meas.add(meas)
            else:
                meas_gt_to_iterate.discard(meas)
        for me_gt in measure_gt:
            if me_gt not in gt_meas_used:
                fn.add(me_gt)

        fact_gt_use = fact_gt.lower()
        fact_generated_use = fact_generated.lower()

        tp = tp.union(tp_meas)
        fn = fn.union(fn_meas)
        fp = fp.union(fp_meas)

        if fact_gt_use == fact_generated_use:
            tp.add(fact_generated)
        else:
            fn.add(fact_gt)
            fp.add(fact_generated)

        # Remove intersection
        fn -= tp
        fp -= tp

        return tp, fp, fn

    def get_nodes(self):
        return self._get_nodes(self.gt_nodes_set, self.out_nodes_set,
                                                      self.gt_preprocessed['measures'],
                                                      self.out_preprocessed['measures'], self.gt_preprocessed['fact'],
                                                      self.out_preprocessed['fact'])


    def calculate_metrics_nodes(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        tp_nodes, fp_nodes, fn_nodes = self.get_nodes()

        precision_nodes, recall_nodes, f1_nodes, tp_nodes, fn_nodes, fp_nodes = _calc_metrics(tp_nodes, fn_nodes, fp_nodes)

        decimals = 4
        return {
            'tp': tp_nodes,
            'fn': fn_nodes,
            'fp': fp_nodes,
            'precision': round(precision_nodes, decimals),
            'recall': round(recall_nodes, decimals),
            'f1': round(f1_nodes, decimals),
        }

    def calculate_metrics_from_preprocessed(self, tp, fp, fn):
        precision_nodes, recall_nodes, f1_nodes, tp_nodes, fn_nodes, fp_nodes = _calc_metrics(tp, fp, fn)
        decimals = 4
        return {
            'tp': tp_nodes,
            'fn': fn_nodes,
            'fp': fp_nodes,
            'precision': round(precision_nodes, decimals),
            'recall': round(recall_nodes, decimals),
            'f1': round(f1_nodes, decimals),
        }


    def calculate_metrics_from_preprocessed_edges(self, tp_idx, fp_idx, fn_idx):
        precision_edges, recall_edges, f1_edges, tp_edges, fn_edges, fp_edges = _calc_metrics(tp_idx, fp_idx, fn_idx)
        decimals = 4
        return {
            'tp': tp_edges,
            'fn': fn_edges,
            'fp': fp_edges,
            'precision': round(precision_edges, decimals),
            'recall': round(recall_edges, decimals),
            'f1': round(f1_edges, decimals),
        }

    def _preprocess(self):
        (self.gt_preprocessed['dependencies'],
         self.gt_preprocessed['measures'],
         self.gt_preprocessed['fact'],
         self.gt_edges_list,
         self.gt_nodes_set) = self._calc_preprocess(self.gt_raw['dependencies'], self.gt_raw['measures'], self.gt_raw['fact'])
        (self.out_preprocessed['dependencies'],
         self.out_preprocessed['measures'],
         self.out_preprocessed['fact'],
         self.out_edges_list,
         self.out_nodes_set) = self._calc_preprocess(self.out_raw['dependencies'], self.out_raw['measures'],
                                                    self.out_raw['fact'])

    def _calc_preprocess(self, dependencies, measures, fact):

        dep_preprocessed = []

        for d in dependencies:
            lab_list = [set(v.split(',')) for v in [d['from'], d['to']]]
            if 'role' in d:
                lab_list.append({d['role']})
            dep_preprocessed.append(lab_list)

        meas_preprocessed = {v for d in measures for _, v in d.items()}
        fact_preprocessed = fact['name']
        edges_list = dep_preprocessed
        nodes_set = _load_nodes(edges_list)

        return dep_preprocessed, meas_preprocessed, fact_preprocessed, edges_list, nodes_set

    def _calculate_edges_indexes(self):
        tp_idx, fn_idx, fp_idx, gt_used = set(), set(), set(), set()

        gt_to_iterate = [gt_to_use for gt_to_use in self.gt_edges_list]

        for idx_out, edges_list_out in enumerate(self.out_edges_list):

            inserted = False
            elem_removed = None

            for edges_list_gt in gt_to_iterate:
                set_out_to_use = [{node.lower() for node in subset} for subset in edges_list_out]
                set_gt_to_use = [{node.lower() for node in subset} for subset in edges_list_gt]
                if (set_out_to_use[0] == set_gt_to_use[0] and set_out_to_use[1] == set_gt_to_use[1] and
                        ((len(set_out_to_use) == 2 and len(set_gt_to_use) == 2) or
                         ((len(set_out_to_use) == 3 and len(set_gt_to_use) == 3) and set_out_to_use[2] == set_gt_to_use[2]))):
                    idx_gt = self.gt_edges_list.index(edges_list_gt)
                    gt_used.add(idx_gt)
                    inserted = True
                    elem_removed = edges_list_gt
                    tp_idx.add(idx_out)
                    break
            if not inserted:
                fp_idx.add(idx_out)
            else:
                gt_to_iterate.remove(elem_removed)

        for idx, edges_list_gt in enumerate(self.gt_edges_list):
            if idx not in gt_used:
                fn_idx.add(idx)

        return tp_idx, fp_idx, fn_idx, gt_used

if __name__ == '__main__':

    # Load config
    input_config = load_yaml_from_resources('metrics-config')

    parser = argparse.ArgumentParser(description="Process some configuration.")
    parser.add_argument('--exercise', help='Exercise to use')
    parser.add_argument('--exercise_gt', help='Exercise gt to use')
    parser.add_argument('--dir', help='Directory containing ex inside output')
    parser.add_argument('--demand', help='State if exercise is demand driven')
    args = parser.parse_args()

    ex_config = input_config['exercise']

    if args.exercise:
        ex_config['name'] = args.exercise
    if args.exercise_gt:
        ex_config['gt'] = args.exercise_gt
    if args.dir:
        ex_config['dir'] = args.dir
    if args.demand:
        ex_config['demand'] = args.demand
    # Load exercise
    ex_output = load_output_exercise(ex_config['dir'], ex_config['name'])

    # Calculate metrics

    if 'gt_preprocessed' in ex_output:
        ground_truth = ex_output['gt_preprocessed']
    else:
        print('Calculating ground truth preprocess')
        ground_truth = load_ground_truth_exercise(ex_config['gt'])
        if ex_config['demand']:
            ground_truth = ground_truth['demand_driven']
        else:
            ground_truth = ground_truth['supply_driven']

        # Extracting ex number as last digit in exercise name
        ex_num = extract_ex_num(ex_config['name'])

        ground_truth['dependencies'], ground_truth['measures'], ground_truth['fact'] = preprocess(ex_num, ground_truth['dependencies'],
                                                                                   ground_truth['measures'] if
                                                                                   ground_truth['measures'] else set(),
                                                                                   ground_truth['fact'], ex_config['demand'])

    metrics = []

    dep_gt = ground_truth['dependencies']
    meas_gt = ground_truth['measures'] if ground_truth['measures'] else set()
    fact_gt = ground_truth['fact']

    # Extracting ex number as last digit in exercise name
    ex_num = extract_ex_num(ex_config['name'])

    metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt, ex_num, ex_config['demand'])

    outputs_to_use = []

    if isinstance(ex_output, dict):
        if 'output_preprocessed' in ex_output:
            outputs_to_use = ex_output['output_preprocessed']
        else:
            print('Calculating output preprocess')
            if 'output' in ex_output:
                output_non_preprocessed = ex_output['output']
            else:
                output_non_preprocessed = ex_output

            for output in output_non_preprocessed:
                try:
                    dep_output, meas_output, fact_output = preprocess(ex_num, output['dependencies'],
                                                                      output['measures'] if output[
                                                                          'measures'] else set(),
                                                                      output['fact'], ex_config['demand'])
                    outputs_to_use.append({'dependencies': dep_output, 'measures': meas_output, 'fact': fact_output})
                except:
                    print(f"Output not correctly generated, skipped")
    output_to_save = []
    for i, output in enumerate(outputs_to_use):
        try:
            dep_output, meas_output, fact_output = output['dependencies'], output['measures'] if output['measures'] else set(), output['fact']

            edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used = metric_calc.get_edges_idx(fact_output, meas_output, dep_output)
            tp_nodes, fp_nodes, fn_nodes = metric_calc.get_nodes()

            step_metric = {'edges': metric_calc.calculate_metrics_from_preprocessed_edges(edges_tp_idx, edges_fp_idx, edges_fn_idx),
                           'nodes': metric_calc.calculate_metrics_nodes(fact_output, meas_output, dep_output)}
            metrics.append(step_metric)

            out, gt = label_edges(outputs_to_use[i], ground_truth, edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used)

            output_to_save.append({'dependencies': out['dependencies'], 'fact': out['fact'], 'measures': out['measures'],
                                   'ground_truth_labels': gt, 'nodes': {'tp': list(tp_nodes), 'fp': list(fp_nodes),
                                                                        'fn': list(fn_nodes)}})
        except:
            traceback.print_exc()
            metrics.append({})
            print(f"Output {i}-th not correctly generated, skipped")

    props = dict()
    props['gt_preprocessed'] = ground_truth
    props['output_preprocessed'] = output_to_save
    props['metrics'] = metrics
    store_additional_properties(ex_config['dir'], ex_config['name'], props)
