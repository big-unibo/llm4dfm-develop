import argparse
from pathlib import Path
from tqdm import tqdm
import traceback
import time
import re

from llm4dfm.pipeline.models import Model
from llm4dfm.pipeline.preprocess import preprocess
from llm4dfm.pipeline.utils import (load_yaml_from_resources, load_prompts, store_output, load_ground_truth_exercise, store_csv,
                                    get_timestamp, output_as_valid_yaml, get_dir_label_name, extract_ex_num, label_edges)
from llm4dfm.pipeline.metrics import MetricsCalculator, ErrorDetector

parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--n_runs', help='Number of runs to execute')
parser.add_argument('--exercise', help='Exercise to use')
parser.add_argument('--exercise_num', help='Exercise number to use')
parser.add_argument('--p_version', help='Prompt version to use')
parser.add_argument('--exercise_version', help='Exercise version to use')
parser.add_argument('--model', help='Model used')
parser.add_argument('--model_label', help='Model label to use')
parser.add_argument('--dir_label', help='Directory label to use')

args = parser.parse_args()

if any(value is not None for value in vars(args).values()):
    automatic_run = True
else:
    automatic_run = False

config = load_yaml_from_resources('pipeline-config')
key_config = load_yaml_from_resources('credentials')
model_config = config[f'model_{config['use']}']

if config['use'] != 'import' and config['use'] != 'api':
    raise Exception("No models")

# Argument parsing

if args.n_runs:
    n_runs = int(args.n_runs)
else:
    n_runs = 1
if args.exercise:
    if len(args.exercise.split('/')) > 0:
        exercise = args.exercise.split('/')[-1]
    else:
        exercise = args.exercise
    exercise = '-'.join(Path(exercise).stem.split('-')[:-1])
    ex_name = '-'.join(exercise.split('-')[:2])
    config['exercise']['name'] = ex_name
else:
    exercise = '-'.join((config['exercise']['name'], config['exercise']['version']))
if args.exercise_num:
    config['exercise']['number'] = int(args.exercise_num)
else:
    if not config['exercise']['number']:
        print(f'No ex number given, extracting as last digit in {config['exercise']['name']}')
        # Extracting ex number as last digit in exercise name
        config['exercise']['number'] = extract_ex_num(config['exercise']['name'])
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

if model_config['name'] in key_config and config['use'] in key_config[model_config['name']]['key']:
    model_config['key'] = key_config[model_config['name']]['key'][config['use']]
else:
    model_config['key'] = None

# Model loading

model = Model(config['use'], model_config['name'], model_config, model_config['key'], config['debug_prints'],
              model_config['quantization'])

config['output']['dir_label'] = get_dir_label_name(config['exercise']['version'], config['exercise']['prompt_version'], model_config['label'], config['output']['dir_label'])

ex_num = config['exercise']['number']

prompts = load_prompts(config['exercise']['prompt_version'], model_config['name'], exercise)

# Load and preprocess ground-truth

ground_truth = load_ground_truth_exercise(config['exercise']['name'])

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

    for prompt in prompts:

        start_time = time.time()

        # Batch text and prompts
        model_output = model.batch(prompt)

        end_time = time.time()

        elapsed = end_time - start_time

        elapsed_times.append(elapsed)

        model_outputs.append(model_output)

    try:
        model_outputs = output_as_valid_yaml(model_outputs)
    except:
        try:
            # usually output yaml as ```yaml effective_yaml``` so attempt to collect effective_yaml
            model_outputs = output_as_valid_yaml([re.search(r"```(.*?)```", single_output, re.DOTALL).group(1).replace('yaml', '').replace('yml', '') for single_output in model_outputs])
            #print('Yaml collected as ```effective_yaml```')
        except:
            store_output(model_config, config['exercise'], model_outputs, [], {}, config['use'] == 'import', [], [], get_timestamp(), config['output']['dir_label'])
            print("Output not correctly generated, skipped")
            continue

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
            dep_output, meas_output, fact_output = preprocess(ex_num, output['dependencies'],
                                                         output['measures'] if 'measures' in output and output['measures'] else list(),
                                                         output['fact'], is_demand, gt_preprocessed['dependencies'])
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
    store_output(model_config, config['exercise'], model_outputs, output_preprocessed, gt_preprocessed,
                 config['use'] == 'import', metrics, detection_list, ts, config['output']['dir_label'])

    if automatic_run:
        store_csv(model_config, config['exercise'], output_preprocessed, config['use'] == 'import',
                  metrics, detection_list, ts, config['output']['dir_label'], elapsed_times)
