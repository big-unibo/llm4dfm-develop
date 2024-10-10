import argparse
from copy import deepcopy

from utils import load_ground_truth_exercise, load_output_exercise, load_yaml_from_resources, append_metrics, \
    extract_ex_num, label_edges


def _calc_metrics(tp, fn, fp):
    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1, tp_count, fn_count, fp_count


def get_metrics_nodes(dep_gt, dep_generated, measure_gt, measure_generated, fact_gt, fact_generated):
    tp_dep = {dep.lower() for dep in dep_gt & dep_generated}
    fn_dep = {dep.lower() for dep in dep_gt - tp_dep}
    fp_dep = {dep.lower() for dep in dep_generated - tp_dep}

    tp_meas = {mes.lower() for mes in measure_gt & measure_generated}
    fn_meas = {mes.lower() for mes in measure_gt - tp_meas}
    fp_meas = {mes.lower() for mes in measure_generated - tp_meas}

    fact_gt_use = fact_gt.lower()
    fact_generated_use = fact_generated.lower()

    tp = tp_dep.union(tp_meas)
    fn = fn_dep.union(fn_meas)
    fp = fp_dep.union(fp_meas)

    if fact_gt_use == fact_generated_use:
        tp.add(fact_gt_use)
    else:
        fn.add(fact_gt_use)
        fp.add(fact_generated_use)

    # Remove intersection
    fn -= tp
    fp -= tp

    # print(f'\nNodes\n\nTP: {tp}\n\nLenTP: {len(tp)}\n\nFN: {fn}\n\nLenFN: {len(fn)}\n\nFP: {fp}\n\nLenFP: {len(fp)}\n\n\n\n')

    return _calc_metrics(tp, fn, fp)


def get_edges_indexes(gt, out):
    tp_idx, fn_idx, fp_idx = set(), set(), set()

    gt_used = set()

    for idx_out, edges_list_out in enumerate(out):
        inserted = False
        for idx_gt, edges_list_gt in enumerate(gt):
            if edges_list_out[0] == edges_list_gt[0] and edges_list_out[1] == edges_list_gt[1]:
                gt_used.add(idx_gt)
                inserted = True
                tp_idx.add(idx_out)
                break
        if not inserted:
            fp_idx.add(idx_out)

    for idx, edges_list_gt in enumerate(gt):
        if idx not in gt_used:
            fn_idx.add(idx)

    return tp_idx, fp_idx, fn_idx, gt_used


def get_metrics_edges(gt, generated):
    tp, fn, fp = [], [], []
    tp_idx, fn_idx, fp_idx = set(), set(), set()

    gt_used = set()

    for idx_out, edges_list_out in enumerate(generated):
        inserted = False
        for idx_gt, edges_list_gt in enumerate(gt):
            if edges_list_out[0] == edges_list_gt[0] and edges_list_out[1] == edges_list_gt[1]:
                gt_used.add(idx_gt)
                inserted = True
                tp_idx.add(idx_out)
                tp.append([edges_list_out[0], edges_list_out[1]])
                break
        if not inserted:
            fp_idx.add(idx_out)
            fp.append([edges_list_out[0], edges_list_out[1]])

    for idx, edges_list_gt in enumerate(gt):
        if idx not in gt_used:
            fn.append([edges_list_gt[0], edges_list_gt[1]])
            fn_idx.add(idx)

    # print(generated, ground_truth)
    # print(tp_idx, fp_idx, fn_idx, gt_used)

    return _calc_metrics(tp, fn, fp)

def _load_nodes(dependencies):
    return set(node for sublist in dependencies for subset in sublist for node in subset)

class MetricsCalculator:

    def __init__(self, gt_fact, gt_measures, gt_dependencies, ex_number, demand=False):
        self.gt_raw = dict()
        self.out_raw = dict()
        self.gt_preprocessed = dict()
        self.gt_edges_set = set()
        self.gt_nodes_set = set()
        self.out_preprocessed = dict()
        self.out_edges_set = set()
        self.out_nodes_set = set()
        self.ex_number = ex_number
        self._load_gt(gt_fact, gt_measures, gt_dependencies)
        self.demand = demand

    def _load_gt(self, gt_fact, gt_measures, gt_dependencies):
        self.gt_raw['fact'] = gt_fact
        self.gt_raw['measures'] = gt_measures
        self.gt_raw['dependencies'] = gt_dependencies

    def _load_gt_ex(self, gt_fact, gt_measures, gt_dependencies, ex_number):
        self.gt_raw['fact'] = gt_fact
        self.gt_raw['measures'] = gt_measures
        self.gt_raw['dependencies'] = gt_dependencies
        self.ex_number = ex_number

    def calculate_metrics(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        precision_edges, recall_edges, f1_edges, tp_edges, fn_edges, fp_edges = get_metrics_edges(self.gt_edges_set,
                                                                                                  self.out_edges_set)
        precision_nodes, recall_nodes, f1_nodes, tp_nodes, fn_nodes, fp_nodes = get_metrics_nodes(self.gt_nodes_set,
                                                                                                  self.out_nodes_set,
                                                                                                  self.gt_preprocessed['measures'],
                                                                                                  self.out_preprocessed['measures'],
                                                                                                  self.gt_preprocessed['fact'],
                                                                                                  self.out_preprocessed['fact'])
        decimals = 4
        return {
            'edges': {
                'tp': tp_edges,
                'fn': fn_edges,
                'fp': fp_edges,
                'precision': round(precision_edges, decimals),
                'recall': round(recall_edges, decimals),
                'f1': round(f1_edges, decimals),
            },
            'nodes': {
                'tp': tp_nodes,
                'fn': fn_nodes,
                'fp': fp_nodes,
                'precision': round(precision_nodes, decimals),
                'recall': round(recall_nodes, decimals),
                'f1': round(f1_nodes, decimals),
            }
        }

    def calculate_metrics_nodes(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        precision_nodes, recall_nodes, f1_nodes, tp_nodes, fn_nodes, fp_nodes = get_metrics_nodes(self.gt_nodes_set,
                                                                                                  self.out_nodes_set,
                                                                                                  self.gt_preprocessed['measures'],
                                                                                                  self.out_preprocessed['measures'],
                                                                                                  self.gt_preprocessed['fact'],
                                                                                                  self.out_preprocessed['fact'])
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

    def calculate_idx(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        return get_edges_indexes(self.gt_edges_set, self.out_edges_set)


    def _preprocess(self):
        (self.gt_preprocessed['dependencies'],
         self.gt_preprocessed['measures'],
         self.gt_preprocessed['fact'],
         self.gt_edges_set,
         self.gt_nodes_set) = self._calc_preprocess(self.gt_raw['dependencies'], self.gt_raw['measures'], self.gt_raw['fact'])
        (self.out_preprocessed['dependencies'],
         self.out_preprocessed['measures'],
         self.out_preprocessed['fact'],
         self.out_edges_set,
         self.out_nodes_set) = self._calc_preprocess(self.out_raw['dependencies'], self.out_raw['measures'],
                                                    self.out_raw['fact'])

    def _calc_preprocess(self, dependencies, measures, fact):
        dep_preprocessed = [[set(v.split(',')) for v in d.values()] for d in dependencies]
        meas_preprocessed = {v.lower() for d in measures for _, v in d.items()}
        fact_preprocessed = fact['name'].lower()
        edges_set = dep_preprocessed
        nodes_set = _load_nodes(edges_set)

        return dep_preprocessed, meas_preprocessed, fact_preprocessed, edges_set, nodes_set

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
        print('using standard ground truth')
        ground_truth = load_ground_truth_exercise(ex_config['gt'])
        if ex_config['demand']:
            ground_truth = ground_truth['demand_driven']
        else:
            ground_truth = ground_truth['supply_driven']

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
            print('using standard output')
            if 'output' in ex_output:
                outputs_to_use = ex_output['output']
            else:
                if isinstance(ex_output, list):
                    outputs_to_use = ex_output
                else:
                    outputs_to_use.append(ex_output)

    for i, output in enumerate(deepcopy(outputs_to_use)):
        try:
            dep_output, meas_output, fact_output = output['dependencies'], output['measures'] if output['measures'] else set(), output['fact']

            work_with_index = False

            if work_with_index:
                print(output['dependencies'])

                tp_idx, fp_idx, fn_idx, gt_used = metric_calc.calculate_idx(fact_output, meas_output, dep_output)
                step_metric = {'edges': metric_calc.calculate_metrics_from_preprocessed_edges(tp_idx, fp_idx, fn_idx),
                               'nodes': metric_calc.calculate_metrics_nodes(fact_output, meas_output, dep_output)}
                metrics.append(step_metric)

                for idx, dep in outputs_to_use[i]['dependencies']:
                    if idx in tp_idx:
                        dep['label'] = 'tp'
                    elif idx in fp_idx:
                        dep['label'] = 'fp'
                    else:
                        dep['label'] = 'error'

                label_edges(outputs_to_use[i], ground_truth, tp_idx, fp_idx, fn_idx, gt_used)
            else:
                metrics.append(metric_calc.calculate_metrics(fact_output, meas_output, dep_output))
        except Exception as e:
            print(e)
            metrics.append({})
            print(f"Output {i}-th not correctly generated, skipped")

    append_metrics(ex_config['dir'], ex_config['name'], metrics)
