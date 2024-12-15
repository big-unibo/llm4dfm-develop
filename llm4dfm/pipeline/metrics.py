import argparse
import traceback
import re

from llm4dfm.pipeline.utils import load_ground_truth_exercise, load_output_exercise, load_yaml_from_resources, \
    extract_ex_num, label_edges, store_additional_properties, update_csv
from collections import defaultdict
from llm4dfm.pipeline.preprocess import preprocess

def _calc_metrics(tp, fp, fn):
    tp_count = len(tp)
    fn_count = len(fn)
    fp_count = len(fp)

    precision = tp_count / (tp_count + fp_count) if tp_count + fp_count > 0 else 0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0

    return precision, recall, f1, tp_count, fn_count, fp_count

def _load_nodes(dependencies):
    return set(','.join(subset) for sublist in dependencies for subset in sublist[:2])


# Computation split in calculate idx and then get metrics to label edges and nodes in output file
class MetricsCalculator:

    def __init__(self, gt_fact, gt_measures, gt_dependencies, ex_number='', demand=False):
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

    def _get_deps_lowercase(self, deps):
        dep_to_iterate = set()
        dep_to_iterate_cache_lowercase = set()

        for dep in deps:
            if dep.lower() not in dep_to_iterate_cache_lowercase:
                dep_to_iterate_cache_lowercase.add(dep.lower())
                dep_to_iterate.add(dep)

        return dep_to_iterate

    def _get_nodes(self, deps_gt, dep_generated, measure_gt, measure_generated, fact_gt, fact_generated):

        dep_gt_to_iterate = self._get_deps_lowercase(deps_gt)
        dep_gen_to_iterate = self._get_deps_lowercase(dep_generated)

        # Starting from fact, then measures, to correctly classify dependencies attributes considering as fact or measures

        tp, fn, fp, gt_used = set(), set(), set(), set()
        fn_cache_lowercase = dict()
        meas_or_fact_wrong = set()

        fact_gt_use = fact_gt.lower()
        fact_generated_use = fact_generated.lower()

        if fact_gt_use == fact_generated_use:
            tp.add(fact_generated_use)
        else:
            fn.add(fact_gt_use)
            fn_cache_lowercase[fact_gt_use] = fact_gt
            fp.add(fact_generated_use)
            meas_or_fact_wrong.add(fact_generated_use)
            meas_or_fact_wrong.add(fact_gt_use)

        meas_gt_to_iterate = self._get_deps_lowercase(measure_gt)
        meas_gen_to_iterate = self._get_deps_lowercase(measure_generated)

        gt_meas_used = set()

        for meas in meas_gen_to_iterate:
            inserted = False
            for meas_gt in meas_gt_to_iterate:
                if meas.lower() == meas_gt.lower():
                    inserted = True
                    tp.add(meas.lower())
                    gt_meas_used.add(meas)
                    break
            if not inserted:
                fp.add(meas.lower())
                meas_or_fact_wrong.add(meas.lower())
            else:
                meas_gt_to_iterate.discard(meas)
        for me_gt in meas_gt_to_iterate:
            if me_gt not in gt_meas_used:
                fn.add(me_gt.lower())
                fn_cache_lowercase[me_gt.lower()] = me_gt
                meas_or_fact_wrong.add(me_gt.lower())

        for dep in dep_gen_to_iterate:
            inserted = False
            if dep.lower() in fn_cache_lowercase or dep.lower() in meas_or_fact_wrong:
                inserted = True
                fp.add(dep.lower())
            if not inserted:
                for dp_gt in dep_gt_to_iterate:
                    if {dp.lower() for dp in dep.split(',')} == {dp.lower() for dp in dp_gt.split(',')}:
                        inserted = True
                        tp.add(dep.lower())
                        gt_used.add(dp_gt)
                        break
                if not inserted:
                    fp.add(dep.lower())
                else:
                    dep_gt_to_iterate.discard(dep)
        for dp_gt in dep_gt_to_iterate:
            if dp_gt not in gt_used:
                fn.add(dp_gt.lower())
                # Used to remove it in case it's present in measures set or fact
                fn_cache_lowercase[dp_gt.lower()] = dp_gt

        fn -= tp
        fp -= tp

        return tp, fp, fn

    def get_nodes(self):
        return self._get_nodes(self.gt_nodes_set, self.out_nodes_set,
                                                      self.gt_preprocessed['measures'],
                                                      self.out_preprocessed['measures'], self.gt_preprocessed['fact'],
                                                      self.out_preprocessed['fact'])

    def calculate_metrics_from_preprocessed(self, tp, fp, fn):
        precision, recall, f1, tp, fn, fp = _calc_metrics(tp, fp, fn)
        decimals = 4
        return {
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'precision': round(precision, decimals),
            'recall': round(recall, decimals),
            'f1': round(f1, decimals),
        }

    def calculate_metrics_nodes(self, out_fact, out_measures, out_dependencies):
        self.out_raw['fact'] = out_fact
        self.out_raw['measures'] = out_measures
        self.out_raw['dependencies'] = out_dependencies

        self._preprocess()

        tp_nodes, fp_nodes, fn_nodes = self.get_nodes()

        return self.calculate_metrics_from_preprocessed(tp_nodes, fp_nodes, fn_nodes)

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
            lab_list = [set(v.replace(' ', '').split(',')) for v in [d['from'], d['to']]]
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


def get_key_from_node_to_avoid_order(node):
    attr_list = [attr.lower() for attr in node.split(',')]
    attr_list.sort()
    return ''.join(attr_list)


def count_reversed_edges(graph1, graph2):
    graph1_edges = set()
    graph2_edges = set()

    for edge in graph1:
        from_key = get_key_from_node_to_avoid_order(edge['from'])
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        graph1_edges.add((from_key, to_key))

    for edge in graph2:
        from_key = get_key_from_node_to_avoid_order(edge['from'])
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        # Put it reversed
        graph2_edges.add((to_key, from_key))

    reversed_edges = graph1_edges & graph2_edges

    return len(reversed_edges)

def count_nodes_with_multiple_incoming_edges(graph):
    nodes_incoming_edges_count = dict()

    for edge in graph:
        to_key = get_key_from_node_to_avoid_order(edge['to'])

        if to_key in nodes_incoming_edges_count:
            nodes_incoming_edges_count[to_key] += 1
        else:
            nodes_incoming_edges_count[to_key] = 1

    count = sum(1 for value in nodes_incoming_edges_count.values() if value > 1)

    return count

def count_nodes_with_multiple_incoming_edges_with_fact_root(graph, fact):
    nodes_incoming_edges_count = dict()

    for edge in graph:
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        from_key = get_key_from_node_to_avoid_order(edge['from'])

        count_reachables = 1 if is_reachable(graph, fact, from_key) else 0

        if to_key in nodes_incoming_edges_count:
            a, b = nodes_incoming_edges_count[to_key]
            nodes_incoming_edges_count[to_key] = a+1, b+count_reachables
        else:
            nodes_incoming_edges_count[to_key] = 1, count_reachables

    count = sum(1 for key, (v1,v2) in nodes_incoming_edges_count.items() if v1 > 1 and v1 == v2)

    return count

def is_reachable(graph, start, target):
    """
    Determine if `target` is reachable from `start` in the given graph.

    Args:
        graph (list of dicts): List of edges, where each edge is a dict with 'from' and 'to' keys.
        start (str): The starting node.
        target (str): The target node.

    Returns:
        bool: True if the target is reachable from the start, False otherwise.
    """
    # Create adjacency list representation of the graph
    adjacency_list = defaultdict(list)

    start_key = get_key_from_node_to_avoid_order(start)
    target_key = get_key_from_node_to_avoid_order(target)

    for edge in graph:
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        from_key = get_key_from_node_to_avoid_order(edge['from'])
        adjacency_list[from_key].append(to_key)
        adjacency_list[to_key].append(from_key)

    # Perform DFS to check reachability
    visited = set()
    stack = [start_key]

    while stack:
        current = stack.pop()
        if current == target_key:
            return True
        if current not in visited:
            visited.add(current)
            stack.extend(adjacency_list[current])

    return False

def count_connected_components(graph):
    # Create adjacency list representation of the graph
    adjacency_list = defaultdict(list)
    for edge in graph:
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        from_key = get_key_from_node_to_avoid_order(edge['from'])
        adjacency_list[from_key].append(to_key)
        adjacency_list[to_key].append(from_key)

    visited = set()

    def dfs(sing_node):
        stack = [sing_node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(adjacency_list[current])

    # Count connected components
    connected_components = 0
    for node in adjacency_list:
        if node not in visited:
            connected_components += 1
            dfs(node)

    return connected_components

def count_false_facts(fact, graph):
    from_nodes = set()
    to_nodes = set()

    for edge in graph:
        from_key = get_key_from_node_to_avoid_order(edge['from'])
        to_key = get_key_from_node_to_avoid_order(edge['to'])
        from_nodes.add(from_key)
        to_nodes.add(to_key)

    false_facts = len(from_nodes - to_nodes)

    if get_key_from_node_to_avoid_order(fact) in from_nodes:
        false_facts -= 1

    return false_facts


def has_extra_tags(graph_gt, graph_out):
    roles_gt = set()
    roles_out = set()

    for edge in graph_gt:
        if 'role' in edge and edge['role'].lower() not in roles_gt:
            roles_gt.add(edge['role'].lower())

    for edge in graph_out:
        if 'role' in edge and edge['role'].lower() not in roles_out:
            roles_out.add(edge['role'].lower())

    return len(roles_out) > len(roles_out & roles_gt)


class ErrorDetector:

    def __init__(self, gt_fact, gt_measures, gt_dependencies):
        self.gt_fact = gt_fact
        self.gt_measures = gt_measures
        self.gt_dependencies = gt_dependencies

    def detect(self, out_fact, out_measures, out_dependencies):
        metric_calculator = MetricsCalculator(self.gt_fact, self.gt_measures, self.gt_dependencies)

        _, edges_fp, edges_fn, _ = metric_calculator.get_edges_idx(out_fact, out_measures, out_dependencies)

        return self.detect_with_metrics(out_fact, out_measures, out_dependencies, len(edges_fn), len(edges_fp))

    def detect_with_metrics(self, out_fact, out_measures, out_dependencies, dep_fn, dep_fp):
        dependencies_detection = dict()
        dependencies_detection['reversed'] = count_reversed_edges(self.gt_dependencies, out_dependencies)
        dependencies_detection['missing'] = dep_fn
        dependencies_detection['extra'] = dep_fp

        measures_detection = dict()
        meas_gt_to_use = {meas['name'].lower() for meas in self.gt_measures}
        meas_out_to_use = {meas['name'].lower() for meas in out_measures}
        measures_detection['missing'] = len(meas_gt_to_use) - len(meas_gt_to_use & meas_out_to_use)
        # Second part to add duplicates if presents
        measures_detection['extra'] = len(meas_out_to_use) - len(meas_gt_to_use & meas_out_to_use) + (len(out_measures) - len(meas_out_to_use))

        fact_detection = dict()
        fact_detection['incorrect'] = self.gt_fact['name'].lower() != out_fact['name'].lower()
        fact_detection['false_fact'] = count_false_facts(self.gt_fact['name'], out_dependencies)

        shared_count_gt = count_nodes_with_multiple_incoming_edges(self.gt_dependencies)
        shared_count_out = count_nodes_with_multiple_incoming_edges(out_dependencies)

        shared_with_same_root_count_gt = count_nodes_with_multiple_incoming_edges_with_fact_root(self.gt_dependencies, self.gt_fact['name'])
        shared_with_same_root_count_out = count_nodes_with_multiple_incoming_edges_with_fact_root(out_dependencies, out_fact['name'])

        diff = shared_count_gt - shared_count_out
        if diff > 0:
            missing, extra = diff, 0
        else:
            missing, extra = 0, abs(diff)

        shared_with_same_root_diff = shared_with_same_root_count_gt - shared_with_same_root_count_out
        if shared_with_same_root_diff > 0:
            shared_with_same_root_missing, shared_with_same_root_extra = shared_with_same_root_diff, 0
        else:
            shared_with_same_root_missing, shared_with_same_root_extra = 0, abs(shared_with_same_root_diff)

        attributes_detection = dict()
        attributes_detection['shared_missing'] = missing
        attributes_detection['shared_extra'] = extra
        attributes_detection['shared_with_fact_root_missing'] = shared_with_same_root_missing
        attributes_detection['shared_with_fact_root_extra'] = shared_with_same_root_extra

        miscellaneous_detection = dict()
        miscellaneous_detection['extra_disconnected_components'] = max(count_connected_components(out_dependencies)-1, 0)
        miscellaneous_detection['extra_tags'] = has_extra_tags(self.gt_dependencies, out_dependencies)

        return dependencies_detection, measures_detection, fact_detection, attributes_detection, miscellaneous_detection

if __name__ == '__main__':

    # Load config
    input_config = load_yaml_from_resources('metrics-config')

    parser = argparse.ArgumentParser(description="Process some configuration.")
    parser.add_argument('--exercise', help='Exercise to use')
    parser.add_argument('--exercise_num', help='Exercise number to use')
    parser.add_argument('--exercise_gt', help='Exercise gt to use')
    parser.add_argument('--dir', help='Directory containing ex inside output')
    parser.add_argument('--version', help='Exercise version to use')
    args = parser.parse_args()

    ex_config = input_config['exercise']

    if args.exercise:
        ex_config['name'] = args.exercise
    if args.exercise_num:
        ex_config['number'] = int(args.exercise_num)
    else:
        if not ex_config['number']:
            print(f'No ex_num given, extracting as last digit of {ex_config["name"]}')
            # Extracting ex number as last digit in exercise name
            ex_config['number'] = extract_ex_num(ex_config['name'])
    if args.exercise_gt:
        ex_config['gt'] = args.exercise_gt
    if args.dir:
        ex_config['dir'] = args.dir
    if args.version:
        ex_config['version'] = args.version
    # Load exercise
    ex_output = load_output_exercise(ex_config['dir'], ex_config['name'])

    # Calculate metrics

    # Regular expression to match the timestamp in YYYY-MM-DDTHH-mm-SS format
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", ex_config['name'])
    if match:
        timestamp = match.group()
    else:
        timestamp = ex_config['name']

    is_demand = ex_config['version'].lower() == 'demand'

    if 'gt_preprocessed' in ex_output:
        ground_truth = ex_output['gt_preprocessed']
    else:
        ground_truth = load_ground_truth_exercise(ex_config['gt'])

        if is_demand:
            ground_truth = ground_truth['demand_driven']
        else:
            ground_truth = ground_truth['supply_driven']

        ground_truth['dependencies'], ground_truth['measures'], ground_truth['fact'] = preprocess(ex_config['number'], ground_truth['dependencies'],
                                                                                    ground_truth['measures'] if
                                                                                    ground_truth['measures'] else list(),
                                                                                    ground_truth['fact'], is_demand, list())

    metrics = []

    dep_gt = ground_truth['dependencies']
    meas_gt = ground_truth['measures'] if ground_truth['measures'] else list()
    fact_gt = ground_truth['fact']

    metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt, ex_config['number'], is_demand)
    detector = ErrorDetector(fact_gt, meas_gt, dep_gt)

    outputs_to_use = []

    if isinstance(ex_output, dict):
        if 'output_preprocessed' in ex_output and ex_output['output_preprocessed'] != []:
            outputs_to_use = ex_output['output_preprocessed']
        else:
            if 'output' in ex_output:
                output_non_preprocessed = ex_output['output']
            else:
                output_non_preprocessed = ex_output

            for output in output_non_preprocessed:
                try:
                    dep_output, meas_output, fact_output = preprocess(ex_config['number'], output['dependencies'],
                                                                      output['measures'] if output[
                                                                          'measures'] else list(),
                                                                      output['fact'], is_demand, ground_truth['dependencies'])

                    outputs_to_use.append({'dependencies': dep_output, 'measures': meas_output, 'fact': fact_output})
                except:
                    traceback.print_exc()
                    print(f"Output not correctly generated, skipped")

    output_to_save = []
    detection_list = list()

    for i, output in enumerate(outputs_to_use):
        try:
            dep_output, meas_output, fact_output = output['dependencies'], output['measures'] if output['measures'] else list(), output['fact']
            edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used = metric_calc.get_edges_idx(fact_output, meas_output, dep_output)
            tp_nodes, fp_nodes, fn_nodes = metric_calc.get_nodes()

            step_metric = {'edges': metric_calc.calculate_metrics_from_preprocessed(edges_tp_idx, edges_fp_idx, edges_fn_idx),
                           'nodes': metric_calc.calculate_metrics_nodes(fact_output, meas_output, dep_output)}
            metrics.append(step_metric)

            detected = dict()
            (detected['dependencies'], detected['measures'], detected['fact'], detected['attributes'],
             detected['miscellaneous']) = detector.detect_with_metrics(fact_output, meas_output, dep_output,
                                                          step_metric['edges']['fn'], step_metric['edges']['fp'])
            detection_list.append(detected)

            output_to_use = {'dependencies': dep_output, 'measures': meas_output, 'fact': fact_output}

            out, gt = label_edges(output_to_use, ground_truth, edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used)

            output_to_save.append({'dependencies': out['dependencies'], 'fact': out['fact'], 'measures': out['measures'],
                                   'ground_truth_labels': gt, 'nodes': {'tp': list(tp_nodes), 'fp': list(fp_nodes),
                                                                        'fn': list(fn_nodes)}})
        except:
            traceback.print_exc()
            metrics.append(dict())
            detection_list.append(dict())
            print(f"Output {i}-th not correctly generated, skipped")
    update_csv(ex_config['dir'], timestamp, ex_config['name'], ex_config['number'], ex_config['version'], output_to_save, metrics, detection_list)
    props = dict()
    props['gt_preprocessed'] = ground_truth
    props['output_preprocessed'] = output_to_save
    props['metrics'] = metrics
    props['errors'] = detection_list
    store_additional_properties(ex_config['dir'], ex_config['name'], props)
