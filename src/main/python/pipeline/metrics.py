import argparse
from pathlib import Path

from utils import load_ground_truth_exercise, load_output_exercise, load_yaml, append_metrics

# Turns a table attribute to first letter capitalized to obtain a uniform comparison between gt and output
def get_clean_table_attribute(table_attr):
    if ',' in table_attr:
        new_val = ''
        for attrs in table_attr.split(','):
            if '.' in attrs:
                attr_split = attrs.split('.')
                val = attr_split[0] + '.' + attr_split[1][0].upper() + attr_split[1][1:]
            else:
                val = attrs.capitalize()
            if new_val == '':
                new_val += val
            else:
                new_val += ',' + val
    else:
        if '.' in table_attr:
            attr_split = table_attr.split('.')
            new_val = attr_split[0] + '.' + attr_split[1][0].upper() + attr_split[1][1:]
        else:
            new_val = table_attr[0].upper() + table_attr[1:]
    return new_val.replace(' ', '')

# Filter dict of ground truth if created
def is_a_valid_dependency(dependency_dict):
    if 'refinements' in dependency_dict:
        return dependency_dict['refinements'] != 'created'
    return True

# Dependencies to consider
def is_a_valid_role_dependency(dependency_key):
    return dependency_key in ['from', 'to']


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

    return _calc_metrics(tp, fn, fp)


def get_metrics_edges(ground_truth, generated):
    tp = ground_truth & generated
    fn = ground_truth - tp
    fp = generated - tp

    return _calc_metrics(tp, fn, fp)


def load_edges(dependency_set):
    return set(
        frozenset((key, get_clean_table_attribute(value))
                  for key, value in d.items() if is_a_valid_role_dependency(key))
        for d in dependency_set if is_a_valid_dependency(d))


def load_nodes(edges_set):
    return set(
        get_clean_table_attribute(entry[1])
        for fr_set in edges_set
        for entry in fr_set
    )


def _calc_preprocess(dependencies, measures, fact):
    dep_preprocessed = [{k.lower(): v.lower() for k, v in d.items()} for d in dependencies]
    meas_preprocessed = {v.lower() for d in measures for _, v in d.items()}
    fact_preprocessed = fact['name'].lower()
    edges_set = load_edges(dep_preprocessed)
    nodes_set = load_nodes(edges_set)

    return dep_preprocessed, meas_preprocessed, fact_preprocessed, edges_set, nodes_set


class MetricsCalculator:

    def __init__(self, gt_fact, gt_measures, gt_dependencies, demand=False):
        self.gt_raw = dict()
        self.out_raw = dict()
        self.gt_preprocessed = dict()
        self.gt_edges_set = set()
        self.gt_nodes_set = set()
        self.out_preprocessed = dict()
        self.out_edges_set = set()
        self.out_nodes_set = set()
        self._load_gt(gt_fact, gt_measures, gt_dependencies)
        self.demand = demand

    def _load_gt(self, gt_fact, gt_measures, gt_dependencies):
        self.gt_raw['fact'] = gt_fact
        self.gt_raw['measures'] = gt_measures
        self.gt_raw['dependencies'] = gt_dependencies

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


    def _preprocess(self):
        (self.gt_preprocessed['dependencies'],
         self.gt_preprocessed['measures'],
         self.gt_preprocessed['fact'],
         self.gt_edges_set,
         self.gt_nodes_set) = _calc_preprocess(self.gt_raw['dependencies'], self.gt_raw['measures'], self.gt_raw['fact'])

        (self.out_preprocessed['dependencies'],
         self.out_preprocessed['measures'],
         self.out_preprocessed['fact'],
         self.out_edges_set,
         self.out_nodes_set) = _calc_preprocess(self.out_raw['dependencies'], self.out_raw['measures'],
                                                    self.out_raw['fact'])

if __name__ == '__main__':

    # Load config
    input_config = load_yaml(f'{Path().absolute()}/../resources/metrics-config.yml')

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
    ground_truth = load_ground_truth_exercise(ex_config['gt'])

    if ex_config['demand']:
        ground_truth = ground_truth['demand_driven']
    else:
        ground_truth = ground_truth['supply_driven']

    metrics = []

    dep_gt = ground_truth['dependencies']
    meas_gt = ground_truth['measures'] if ground_truth['measures'] else set()
    fact_gt = ground_truth['fact']

    metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt)

    outputs_to_use = []

    if isinstance(ex_output, dict):
        if 'output' in ex_output:
            outputs_to_use = ex_output['output']
        else:
            if isinstance(ex_output, list):
                outputs_to_use = ex_output
            else:
                outputs_to_use.append(ex_output)

    for i, output in enumerate(outputs_to_use):
        try:
            dep_output, meas_output, fact_output = output['dependencies'], output['measures'] if output['measures'] else set(), output['fact']
            metrics.insert(i, metric_calc.calculate_metrics(fact_output, meas_output, dep_output))
        except:
            metrics.insert(i, {})
            print(f"Output {i}-th not correctly generated, skipped")

    append_metrics(ex_config['dir'], ex_config['name'], metrics)
