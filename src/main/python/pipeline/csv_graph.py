from pathlib import Path
import argparse
from utils import load_csv, load_yaml, enrich_label
from graph_utils import plot_csv_metrics

parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--exercise_v', help='Exercise version to use')
parser.add_argument('--prompt_version', help='Prompt version to use')
parser.add_argument('--model', help='Model to use')
parser.add_argument('--runs', help='Number of executions done')
parser.add_argument('--label', help='Label to use')
args = parser.parse_args()

# Load config
input_config = load_yaml(f'{Path().absolute()}/../resources/visualisation-config.yml')['csv_graph']

# Check if the --exercise argument is passed
if args.exercise_v:
    input_config['v'] = args.exercise_v
if args.prompt_version:
    input_config['prompt_v'] = args.prompt_version
if args.model:
    input_config['model'] = args.model
if args.runs:
    input_config['runs'] = args.runs
if args.label:
    input_config['label'] = args.label

if input_config['label']:
    input_config['label'] = enrich_label(input_config["label"])
else:
    input_config['label'] = ''
if input_config['runs']:
    input_config['runs'] = f'-{input_config["runs"]}'
else:
    input_config['runs'] = ''

f_name = f'{input_config["label"]}output-{input_config['v']}-{input_config["prompt_v"]}-{input_config["model"]}{input_config["runs"]}'

try:
    csv_file = load_csv(f_name)
except:
    print(f'File {f_name}.csv not found')
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

plot_csv_metrics(metrics, f_name)
