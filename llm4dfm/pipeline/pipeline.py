import argparse
from pathlib import Path
from tqdm import tqdm
import traceback
import time

from llm4dfm.pipeline.models import Model, load_text_and_first_prompt, is_model_without_chat_constraints
from llm4dfm.pipeline.preprocess import preprocess
from llm4dfm.pipeline.utils import (load_yaml_from_resources, load_prompts, store_output, load_ground_truth_exercise, store_csv,
                                    get_timestamp, output_as_valid_yaml, get_dir_label_name, extract_ex_num, label_edges, load_text_exercise)
from llm4dfm.pipeline.metrics import MetricsCalculator, ErrorDetector

parser = argparse.ArgumentParser(description="Process some configuration.")
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

model_config = load_yaml_from_resources('pipeline-config')
key_config = load_yaml_from_resources('credentials')
config = model_config[f'model_{model_config['use']}']

if model_config['use'] != 'import' and model_config['use'] != 'api':
    raise Exception("No models")

# Argument parsing

if args.exercise:
    if len(args.exercise.split('/')) > 0:
        exercise = args.exercise.split('/')[-1]
    else:
        exercise = args.exercise
    exercise = '-'.join(Path(exercise).stem.split('-')[:-1])
    ex_name = '-'.join(exercise.split('-')[:2])
    model_config['exercise']['name'] = ex_name
else:
    exercise = '-'.join((model_config['exercise']['name'], model_config['exercise']['version']))
if args.exercise_num:
    model_config['exercise']['number'] = int(args.exercise_num)
else:
    if not model_config['exercise']['number']:
        print(f'No ex number given, extracting as last digit in {model_config['exercise']['name']}')
        # Extracting ex number as last digit in exercise name
        model_config['exercise']['number'] = extract_ex_num(model_config['exercise']['name'])
if args.p_version:
    model_config['exercise']['prompt_version'] = args.p_version
if args.exercise_version:
    model_config['exercise']['version'] = args.exercise_version
if args.dir_label:
    model_config['output']['dir_label'] = args.dir_label
if args.model:
    config['name'] = args.model
if args.model_label:
    config['label'] = args.model_label

config['key'] = key_config[config['name']]['key'][model_config['use']] if model_config['use'] in key_config[config['name']]['key'] else None

# Model loading

model = Model(model_config['use'], config['name'], config, config['key'], model_config['debug_prints'],
              config['quantization'])

model_config['output']['dir_label'] = get_dir_label_name(model_config['exercise']['version'], model_config['exercise']['prompt_version'], config['label'], model_config['output']['dir_label'])

ex_num = model_config['exercise']['number']

# Load prompts

model_outputs = []

### BEGIN - Mode supporting multiple iterations
# prompts = []
# # Load context prompt and then text exercise and first prompt together
# first_prompt = load_text_and_first_prompt(exercise, model_config['exercise']['prompt_version'], config['name'])
# prompts.extend(first_prompt)
# # After, load remaining prompts
# prompts.extend(load_prompts(model_config['exercise']['prompt_version'], config['name'])[len(first_prompt):])

# # Used to allow models without chat structure constraints (i.e. after each system or user input require an assistant
# # message, so one batch at a time) to batch first system and user input in a single batch
# first_batch = len(first_prompt) if is_model_without_chat_constraints(config['name']) else 1

# Batch text and prompts
# with (tqdm(desc=f'Prompt {config["name"]}', total=len(prompts)) as bar_batch):
    # if model_config['debug_prints']:
    #     print(prompts)
    # model_output = model.batch(prompts[:first_batch])
    # model_outputs.append(model_output)
    # bar_batch.update(first_batch)
    # for prompt in prompts[first_batch:]:
    #     model_output = model.batch(prompt)
    #     model_outputs.append(model_output)
    #     bar_batch.update(1)
    

    # model_output = model.batch(prompts)
    # model_outputs.append(model_output)
    # bar_batch.update(1)
### END - Mode supporting multiple iterations

### BEGIN - Mode sending all the conversation in batch
prompts = load_prompts(model_config['exercise']['prompt_version'], config['name'])

prompts[len(prompts)-1]['content'] = "\n".join([prompts[len(prompts)-1]['content'], load_text_exercise(exercise)])

elapsed_times = []

start_time = time.time()

# Batch text and prompts
model_output = model.batch(prompts)

end_time = time.time()  # End the timer
elapsed_times.append(end_time - start_time)

model_outputs.append(model_output)
### END - Mode sending all the conversation in batch

try:
    model_outputs = output_as_valid_yaml(model_outputs)
except:
    store_output(config, model_config['exercise'], model_outputs, [], {}, model_config['use'] == 'import', [], [], get_timestamp(), model_config['output']['dir_label'])
    print("Output not correctly generated")
    exit(1)


if model_config['debug_prints']:
    print(f'Chat: {model.chat}\nOutput: {model_outputs}')

# Load and preprocess ground-truth

ground_truth = load_ground_truth_exercise(model_config['exercise']['name'])

is_demand = model_config['exercise']['version'] == 'demand'

if is_demand:
    ground_truth = ground_truth['demand_driven']
else:
    ground_truth = ground_truth['supply_driven']

# Calculate gt_preprocessed
gt_preprocessed = dict()
gt_preprocessed['dependencies'], gt_preprocessed['measures'], gt_preprocessed['fact'] = preprocess(ex_num, ground_truth['dependencies'],
                                                                                                   ground_truth['measures'] if ground_truth['measures'] else list(),
                                                                                                   ground_truth['fact'], is_demand)

# Calculate metrics

metrics = []

dep_gt = gt_preprocessed['dependencies']
meas_gt = gt_preprocessed['measures']
fact_gt = gt_preprocessed['fact']

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
store_output(config, model_config['exercise'], model_outputs, output_preprocessed, gt_preprocessed,
             model_config['use'] == 'import', metrics, detection_list, ts, model_config['output']['dir_label'])

if automatic_run:
    store_csv(config, model_config['exercise'], output_preprocessed, model_config['use'] == 'import',
              metrics, detection_list, ts, model_config['output']['dir_label'], elapsed_times)
