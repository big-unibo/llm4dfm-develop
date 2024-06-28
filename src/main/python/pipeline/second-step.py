from pathlib import Path
from utils import load_yaml, load_ground_truth_exercise, load_output_exercise
from ssutils import clean_gt_dependencies

input_config = load_yaml(f'{Path().absolute()}/pipeline/second-step-config.yml')

ex_config = input_config['exercise']
model_config = input_config['model']
output_config = input_config['output']

ex_output = load_output_exercise(ex_config['name'], ex_config['v'], ex_config['prompt_v'],
                                 model_config['name'], model_config['v'],
                                 output_config['latest'], output_config['timestamp'], ex_config['full_name'])

ground_truth = load_ground_truth_exercise(input_config['exercise']['name'])

dep_output = ex_output['output'][0]['dependencies']

# Given a different format ground-truth and model-output, uniform it
dep_gt_dirty = ground_truth[0]['dependencies']
dep_gt = clean_gt_dependencies(dep_gt_dirty)

set_gt = set(frozenset(d.items()) for d in dep_gt)
set_output = set(frozenset(d.items()) for d in dep_output)

# Find intersection
common_dicts = set_gt & set_output

tp = len(common_dicts)
fn = len(set_gt) - len(common_dicts)
fp = len(set_output) - len(common_dicts)

print(f"TP: {tp}\nFN: {fn}\nFP: {fp}")

# Optionally, convert frozensets back to dictionaries to see the common elements
common_dicts_list = [dict(fs) for fs in common_dicts]
print("Common dictionaries:", common_dicts_list)
