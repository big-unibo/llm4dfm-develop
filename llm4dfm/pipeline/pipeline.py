import argparse
from pathlib import Path
from tqdm import tqdm
import traceback
import time

from llm4dfm.pipeline.models import Model
from llm4dfm.pipeline.preprocess import preprocess
from llm4dfm.pipeline.utils import (load_yaml_from_resources, store_output, load_ground_truth_exercise,
                                    store_csv,
                                    get_timestamp, output_as_valid_yaml, get_dir_label_name, extract_ex_num,
                                    label_edges, load_prompts_as_single, load_credentials)
from llm4dfm.pipeline.metrics import MetricsCalculator, ErrorDetector

parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--n_runs', help='Number of runs to execute')
parser.add_argument('--exercises', nargs='+', help='List of exercises to use')
parser.add_argument('--p_version', help='Prompt version to use')
parser.add_argument('--exercise_version', help='Exercise version to use')
parser.add_argument('--model', help='Model used')
parser.add_argument('--model_loading', help='Model loading technique used')
parser.add_argument('--model_label', help='Model label to use')
parser.add_argument('--dir_label', help='Directory label to use')
parser.add_argument('--device', help='Set device to use [cpu, gpu]')
parser.add_argument('--debug_print', action='store_true', help='Enable debug prints')

args = parser.parse_args()

if any(value is not None for value in vars(args).values()):
    automatic_run = True
else:
    automatic_run = False

config = load_yaml_from_resources('pipeline-config')
key_config = load_yaml_from_resources('credentials')

# Argument parsing

if args.model_loading:
    config['use'] = args.model_loading

if config['use'] != 'import' and config['use'] != 'api':
    raise Exception(f"No models for {config['use']} use")

model_config = config[f'model_{config['use']}']

if args.n_runs:
    n_runs = int(args.n_runs)
else:
    n_runs = 1
if args.exercises:
    if not type(args.exercises) is list:
        args.exercises = [args.exercises]

    exercises = [ex.split('/')[-1] if len(ex.split('/')) > 0 else ex for ex in args.exercises]

    exercises = ['-'.join(Path(ex).stem.split('-')[:-1]) for ex in exercises]
    ex_name = ['-'.join(ex.split('-')[:2]) for ex in exercises]
    config['exercise']['name'] = ex_name
else:
    config['exercise']['name'] = ['-'.join((ex_name, config['exercise']['version'])) for ex_name in config['exercise']['name']]
if automatic_run or not config['exercise']['number'] or not type(config['exercise']['number']) is list or len(config['exercise']['number']) != len(config['exercise']['name']):
    print(f'Extracting exercise number as last digit in {config['exercise']['name']}')
    # Extracting ex number as last digit in exercise name
    config['exercise']['number'] = [extract_ex_num(ex_name) for ex_name in config['exercise']['name']]
if args.p_version:
    config['exercise']['prompt_version'] = args.p_version
if args.exercise_version:
    config['exercise']['version'] = args.exercise_version
if args.dir_label:
    config['output']['dir_label'] = args.dir_label
if args.model:
    model_config['name'] = args.model
if args.model_label:
    model_config['label'] = args.model_label
if args.device:
    model_config['device'] = args.device
if args.debug_print:
    config['debug_prints'] = True


model_config['key'] = load_credentials(key_config, model_config['name'], config['use'])

# Model loading

model = Model(config['use'], model_config['name'], model_config, model_config['key'], model_config['device'], config['debug_prints'])

config['output']['dir_label'] = get_dir_label_name(config['exercise']['version'], config['exercise']['prompt_version'], model_config['label'], config['use'], config['output']['dir_label'])

for ex_idx, exercise in enumerate(config['exercise']['name']):

    print(f'Execution on {exercise}')

    ex_num = config['exercise']['number'][ex_idx]

    # prompts = load_prompts_as_multiple(config['exercise']['prompt_version'], model_config['name'], '-'.join((exercise, config['exercise']['version'])))
    prompts = load_prompts_as_single(config['exercise']['prompt_version'], model_config['name'], '-'.join((exercise, config['exercise']['version'])))

    # Load and preprocess ground-truth
    ground_truth = load_ground_truth_exercise(exercise)

    is_demand = config['exercise']['version'] == 'demand'

    if is_demand:
        ground_truth = ground_truth['demand_driven']
    else:
        ground_truth = ground_truth['supply_driven']

    # Calculate gt_preprocessed
    gt_preprocessed = dict()
    gt_preprocessed['dependencies'], gt_preprocessed['measures'], gt_preprocessed['fact'] = preprocess(ex_num, ground_truth['dependencies'],
                                                                                                       ground_truth['measures'] if ground_truth['measures'] else list(),
                                                                                                       ground_truth['fact'], is_demand)

    dep_gt = gt_preprocessed['dependencies']
    meas_gt = gt_preprocessed['measures']
    fact_gt = gt_preprocessed['fact']

    for i_run in tqdm(range(n_runs), desc=f"Run"):

        model_outputs = []
        elapsed_times = []
        model.refresh_session()

        start_time = time.time()

        # Batch text and prompts
        model_output = model.batch(prompts)

        end_time = time.time()

        elapsed = end_time - start_time

        elapsed_times.append(elapsed)

        try:
            model_output = output_as_valid_yaml(model_output)
        except:
            print("Output not parsed to yaml, kept as it is")
        model_outputs.append(model_output)

        if config['debug_prints']:
            print(f'Chat: {model.chat}\nOutput: {model_outputs}')

        # Calculate metrics

        metrics = []

        metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt, ex_num, is_demand)
        detector = ErrorDetector(fact_gt, meas_gt, dep_gt)

        output_preprocessed = []
        detection_list = list()

        for i, output in enumerate(model_outputs):
            try:
                # Preprocess output
                dep_output, meas_output, fact_output = preprocess(ex_num, output['dependencies'] if 'dependencies' in output and output['dependencies'] else list(),
                                                             output['measures'] if 'measures' in output and output['measures'] else list(),
                                                             output['fact'] if 'fact' in output and output['fact'] else dict(), is_demand, gt_preprocessed['dependencies'])
                # Get idxes to label edges correctly
                edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used = metric_calc.get_edges_idx(fact_output, meas_output, dep_output)
                tp_nodes, fp_nodes, fn_nodes = metric_calc.get_nodes()

                step_metric = {
                    'edges': metric_calc.calculate_metrics_from_preprocessed(edges_tp_idx, edges_fp_idx, edges_fn_idx),
                    'nodes': metric_calc.calculate_metrics_nodes(fact_output, meas_output, dep_output)}
                metrics.append(step_metric)

                detected = dict()
                (detected['dependencies'], detected['measures'], detected['fact'], detected['attributes'],
                 detected['miscellaneous']) = detector.detect_with_metrics(fact_output, meas_output, dep_output,
                                                              step_metric['edges']['fn'], step_metric['edges']['fp'])
                detection_list.append(detected)

                output_to_use = {'dependencies': dep_output, 'measures': meas_output, 'fact': fact_output}

                out, gt = label_edges(output_to_use, gt_preprocessed, edges_tp_idx, edges_fp_idx, edges_fn_idx, gt_used)

                output_preprocessed.append({'dependencies': out['dependencies'], 'fact': out['fact'], 'measures': out['measures'],
                                       'ground_truth_labels': gt, 'nodes': {'tp': list(tp_nodes), 'fp': list(fp_nodes),
                                                                            'fn': list(fn_nodes)}})
            except:
                traceback.print_exc()
                detection_list.append(dict())
                metrics.insert(i, dict())
                print(f"Output {i}-th not correctly generated, skipped")

        # Store results
        ts = get_timestamp()

        # store output
        store_output(model_config, config['exercise']['prompt_version'], exercise, config['exercise']['version'], model_outputs, output_preprocessed, gt_preprocessed,
                     config['use'] == 'import', metrics, detection_list, ts, config['output']['dir_label'])

        if automatic_run:
            store_csv(model_config, exercise, config['exercise']['version'], config['exercise']['prompt_version'], ex_num, output_preprocessed, config['use'] == 'import',
                      metrics, detection_list, ts, config['output']['dir_label'], elapsed_times)
