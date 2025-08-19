from pathlib import Path
import argparse

from llm4dfm.pipeline.utils import load_full_path_csv, load_yaml_from_resources, get_dir_label_name, get_csv_file_from_output_dir
from llm4dfm.pipeline.graph_utils import plot_csv_metrics, plot_time_f1

parser = argparse.ArgumentParser(description="Process some configuration.")
parser.add_argument('--prompt_version', help='Prompt version to use')
parser.add_argument('--exercise_v', help='Exercise version to use')
parser.add_argument('--model_label', help='Model label to use')
parser.add_argument('--dir_label', help='Directory label to use')
parser.add_argument('--model_loading', help='Model loading technique used')
parser.add_argument('--device', help='Set device used [cpu, gpu]')
parser.add_argument('--dir', help='Full dir name to use')
args = parser.parse_args()

# Load config
input_config = load_yaml_from_resources('csv-graph-config')

file_path = ''

if args.dir:
    input_config['dir_label'] = args.dir
else:
    if args.prompt_version:
        input_config['prompt_v'] = args.prompt_version
    if args.exercise_v:
        input_config['v'] = args.exercise_v
    if args.model_label:
        input_config['model_label'] = args.model_label
    if args.dir_label:
        input_config['dir_label'] = args.dir_label
    if args.model_loading:
        input_config['model_loading'] = args.model_loading
    if args.device:
        input_config['device'] = args.device
    input_config['dir_label'] = get_dir_label_name(input_config['v'], input_config['prompt_v'], input_config['model_label'], input_config['dir_label'], input_config['model_loading'], input_config['device'])
    if not Path(get_csv_file_from_output_dir(input_config['dir_label'])).is_file() and input_config['device'] == 'gpu':
        input_config['dir_label'] = get_dir_label_name(input_config['v'], input_config['prompt_v'],
                                                       input_config['model_label'], input_config['dir_label'],
                                                       input_config['model_loading'], input_config['device'], force_gpu_for_retrieve=True)

file_path = get_csv_file_from_output_dir(input_config['dir_label'])

try:
    csv_file = load_full_path_csv(file_path)
except:
    print(f'CSV file {file_path} not found, no graphs plotted.')
    exit(1)

metrics_template = ['edges_precision', 'edges_recall', 'edges_f1', 'nodes_precision', 'nodes_recall', 'nodes_f1',]
metrics = dict()

for ex_name in csv_file['ex_name'].unique():
    # Filter rows where ex_name matches the current value
    filtered_df = csv_file[csv_file['ex_name'] == ex_name]

    # Create a dictionary with the desired columns as lists
    metrics[ex_name] = {
        'edges_precision': filtered_df['edges_precision'].tolist(),
        'edges_recall': filtered_df['edges_recall'].tolist(),
        'edges_f1': filtered_df['edges_f1'].tolist(),
        'nodes_precision': filtered_df['nodes_precision'].tolist(),
        'nodes_recall': filtered_df['nodes_recall'].tolist(),
        'nodes_f1': filtered_df['nodes_f1'].tolist()
    }

plot_csv_metrics(metrics, Path(file_path).parent, input_config['dir_label'])
plot_time_f1(csv_file, Path(file_path).parent, input_config['dir_label'])
