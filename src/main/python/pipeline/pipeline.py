import os
import argparse
from pathlib import Path
from tqdm import tqdm
from models import Model, load_text_and_first_prompt, is_model_without_chat_constraints
from utils import (load_yaml, load_prompts, store_output, load_ground_truth_exercise, store_automatic_output,
                   get_timestamp, output_as_valid_yaml, get_dir_label_name)
from metrics import MetricsCalculator

def log(message):
    print(f'{os.path.splitext(os.path.basename(__file__))[0]} - {message}\n')


parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--exercise', help='Exercise to use')
parser.add_argument('--p_version', help='Prompt version to use')
parser.add_argument('--exercise_version', help='Exercise version to use')
parser.add_argument('--model', help='Model used')
parser.add_argument('--model_label', help='Model label to use')
parser.add_argument('--dir_label', help='Directory label to use')

args = parser.parse_args()

model_config = load_yaml(f'{Path().absolute()}/../resources/pipeline-config.yml')
key_config = load_yaml(f'{Path().absolute()}/../resources/credentials.yml')

# Model loading

if model_config['use'] == 'import':
    raise Exception('import not supported')
    # config = model_config['model_import']
    #
    # config['key'] = key_config[config['name']]['key']
    #
    # model = Model(model_config['use'], config['name'], config, config['key'],
    #               model_config['debug_prints'], config['quantization'])

elif model_config['use'] == 'api':
    config = model_config['model_api']
    if args.model:
        automatic_run = True
        config['name'] = args.model

    config['key'] = key_config[config['name']]['key']
    model = Model(model_config['use'], config['name'], config, config['key'], model_config['debug_prints'])

else:
    raise Exception("No models")

# Argument parsing

automatic_run = False

# Check if the --exercise argument is passed
if args.exercise:
    automatic_run = True
    if len(args.exercise.split('/')) > 0:
        exercise = args.exercise.split('/')[-1]
    else:
        exercise = args.exercise
    exercise = '-'.join(Path(exercise).stem.split('-')[:-1])
    ex_name = '-'.join(exercise.split('-')[:2])
    model_config['exercise']['name'] = ex_name
else:
    exercise = '-'.join((model_config['exercise']['name'], model_config['exercise']['version']))
if args.p_version:
    automatic_run = True
    model_config['exercise']['prompt_version'] = args.p_version
if args.exercise_version:
    automatic_run = True
    model_config['exercise']['version'] = args.exercise_version
if args.model_label:
    automatic_run = True
    config['label'] = args.model_label
if args.dir_label:
    automatic_run = True
    model_config['output']['dir_label'] = args.dir_label

model_config['output']['dir_label'] = get_dir_label_name(model_config['exercise']['version'], model_config['exercise']['prompt_version'], config['label'], model_config['output']['dir_label'])

# Load prompts

model_outputs = []
prompts = []
# As new indication, load context prompt and then text exercise and first prompt together
first_prompt = load_text_and_first_prompt(exercise, model_config['exercise']['prompt_version'], config['name'])
prompts.extend(first_prompt)
# After, load remaining prompts
prompts.extend(load_prompts(model_config['exercise']['prompt_version'], config['name'])[len(first_prompt):])

# Used to allow models without chat structure constraints (i.e. after each system or user input require an assistant
# message, so one batch at a time) to batch first system and user input in a single batch
first_batch = len(first_prompt) if is_model_without_chat_constraints(config['name']) else 1

# Batch text and prompts

with (tqdm(desc=f'Prompt {config["name"]}', total=len(prompts)) as bar_batch):
    if model_config['debug_prints']:
        print(prompts)
    model_output = model.batch(prompts[:first_batch])
    model_outputs.append(model_output)
    bar_batch.update(first_batch)
    for prompt in prompts[first_batch:]:
        model_output = model.batch(prompt)
        model_outputs.append(model_output)
        bar_batch.update(1)

try:
    model_outputs = output_as_valid_yaml(model_outputs)
except:
    store_output(config, model_config['exercise'], model_outputs, model_config['use'] == 'import', {}, get_timestamp(), model_config['output']['dir_label'])
    print("Output not correctly generated")
    exit(1)


if model_config['debug_prints']:
    log(f'Chat: {model.chat}\nOutput: {model_output}')

# Calculate metrics
ground_truth = load_ground_truth_exercise(model_config['exercise']['name'])

if model_config['exercise']['version'] == 'demand':
    ground_truth = ground_truth['demand_driven']
else:
    ground_truth = ground_truth['supply_driven']

metrics = []

dep_gt = ground_truth['dependencies']
meas_gt = ground_truth['measures'] if ground_truth['measures'] else set()
fact_gt = ground_truth['fact']

metric_calc = MetricsCalculator(fact_gt, meas_gt, dep_gt)

for i, output in enumerate(model_outputs):
    try:
        dep_output, meas_output, fact_output = model_outputs[i]['dependencies'], model_outputs[i]['measures'] if model_outputs[i]['measures'] else set(), model_outputs[i]['fact']
        metrics.insert(i, metric_calc.calculate_metrics(fact_output, meas_output, dep_output))
    except:
        metrics.insert(i, {})
        print(f"Output {i}-th not correctly generated, skipped")

ts = get_timestamp()

# store output
store_output(config, model_config['exercise'], model_outputs, model_config['use'] == 'import', metrics, ts, model_config['output']['dir_label'])

if automatic_run:
    store_automatic_output(config, model_config['exercise'], model_outputs, model_config['use'] == 'import', metrics, ts, model_config['output']['dir_label'])
