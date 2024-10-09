from pathlib import Path
import argparse

from utils import load_full_path_csv, load_yaml_from_resources, get_output_file_path, get_dir_label_name
from graph_utils import plot_csv_metrics

parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--prompt_version', help='Prompt version to use')
parser.add_argument('--exercise_v', help='Exercise version to use')
parser.add_argument('--model_label', help='Model label to use')
parser.add_argument('--dir_label', help='Directory label to use')
args = parser.parse_args()

# Load config
input_config = load_yaml_from_resources('visualisation-config')['csv_graph']

# Check if the --exercise argument is passed
if args.prompt_version:
    input_config['prompt_v'] = args.prompt_version
if args.exercise_v:
    input_config['v'] = args.exercise_v
if args.model_label:
    input_config['model_label'] = args.model_label
if args.dir_label:
    input_config['dir_label'] = args.dir_label

input_config['dir_label'] = get_dir_label_name(input_config['v'], input_config['prompt_v'], input_config['model_label'], input_config['dir_label'])
file_path = get_output_file_path(input_config['v'], input_config['prompt_v'], input_config['model_label'], input_config['dir_label'])

try:
    csv_file = load_full_path_csv(file_path)
except:
    print(f'File {file_path} not found')
    exit(1)

metrics_template = ['edges_precision', 'edges_recall', 'edges_f1', 'nodes_precision', 'nodes_recall', 'nodes_f1',]
metrics = dict()

for row in csv_file:
    ex = row['ex_name']
    if ex not in metrics:
        metrics[ex] = dict()
        for metr_templ in metrics_template:
            metrics[ex][metr_templ] = []
    for metr in metrics[ex]:
        metrics[ex][metr].append(row[metr])

plot_csv_metrics(metrics, Path(file_path).parent, input_config['dir_label'])
