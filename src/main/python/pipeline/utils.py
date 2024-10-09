import os
from dotenv import load_dotenv
import yaml
from datetime import datetime
import re
import csv
from pathlib import Path

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')
auto_outputs = os.getenv('AUTO_OUTPUTS')


def log(message):
    print(f'{os.path.splitext(os.path.basename(__file__))[0]} - {message}\n')

# datetime object containing current date and time
def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

def get_dir_label_name(ex_version, prompt_version, model_label, dir_label):
    return f"{ex_version}-{prompt_version}-{model_label}-{dir_label}"

# return yaml configurations as dict
def load_yaml(yaml_file) -> dict:
    with open(f'{yaml_file}.yml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# return yaml configurations as dict
def load_yaml_from_resources(yaml_filename) -> dict:
    return load_yaml(f'{Path().absolute()}/../resources/{yaml_filename}')

def write_yaml(yaml_file, data):
    with open(f'{yaml_file}.yml', 'w+', encoding='utf-8') as file:
        yaml.dump(data, file)

def load_csv(file_name):
    return load_full_path_csv(f'{outputs}{file_name}.csv')

def load_full_path_csv(path):
    with open(f'{path}', 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]

# return the text of exercise given ex_name
def load_text_exercise(ex_name):
    return load_yaml(f'{datasets}{ex_name}-text')['text']


# return the ground truth of exercise given ex_name
def load_ground_truth_exercise(ex_name, full_name=''):
    if full_name:
        file_name = '-'.join(full_name.split('-')[:2])
    else:
        file_name = ex_name
    return load_yaml(f'{datasets}{file_name}-ground-truth')


# load output exercise used in second-step and its filename (used after to store the image)
def load_output_exercise(dir_name, full_name):
    return load_yaml(f'{outputs}{dir_name}/{full_name}')


# load output exercise used in second-step and its filename (used after to store the image)
def load_output_exercise_and_name(dir_name, ex_name, version, prompt_version, model_name, model_version, latest=True, timestamp='',
                         full_name=''):
    if full_name:
        exercise = full_name + '.yml'
    else:
        # List all files in the directory
        files = os.listdir(f'{outputs}{dir_name}/')

        if latest:
            exercise_pattern = re.compile(
                rf'(?P<ex_name>{ex_name})-'
                rf'(?P<version>{version})-'
                rf'(?P<prompt_version>{prompt_version})-'
                rf'(?P<model_name>{model_name})-'
                rf'(?P<model_version>{model_version})-'
            ) if model_version else re.compile(
                rf'(?P<ex_name>{ex_name})-'
                rf'(?P<version>{version})-'
                rf'(?P<prompt_version>{prompt_version})-'
                rf'(?P<model_name>{model_name})-'
            )

            # Regex pattern to extract date from file name
            date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})')

            # Function to extract date from file name and convert to datetime object
            def extract_date(file_name):
                match = date_pattern.search(file_name)
                if match:
                    date_str = match.group(1)
                    return datetime.strptime(date_str, '%Y-%m-%dT%H-%M-%S')
                return None

            # Initialize variables to track the latest file and its date
            latest_file = None
            latest_date = None

            # Iterate over files and find the latest one based on date
            for file_matching in [file for file in files if exercise_pattern.match(file)]:
                file_date = extract_date(file_matching)
                if file_date:
                    if latest_date is None or file_date > latest_date:
                        latest_date = file_date
                        latest_file = file_matching

            # Output the latest file
            if latest_file:
                exercise = latest_file
            else:
                raise Exception("No valid files found.")
        else:
            if not timestamp:
                raise Exception("No timestamp provided, can't find any file.")
            if model_version:
                exercise = '-'.join((ex_name, version, prompt_version, model_name, model_version, timestamp)) + '.yml'
            else:
                exercise = '-'.join((ex_name, version, prompt_version, model_name, timestamp)) + '.yml'

    return load_yaml(f'{outputs}{dir_name}/{exercise}'), exercise


# return prompts of exercise as a dict given ex_name and model_name
def load_prompts(version, model_name):
    return load_yaml(f'{inputs}prompts-{version}')[model_name]


# Extract exercise number based on last digit of exercise name
def extract_ex_num(ex_name):
    numbers = re.findall(r'\d+', ex_name)  # Find all sequences of digits
    if numbers:
        return int(numbers[-1])
    else:
        print("No exercise numbers in exercise name.")
        return None


def label_edges(out, gt, tp_idx, fp_idx, fn_idx, gt_used):
    return None


# Map output to a valid yaml as dict
def output_as_valid_yaml(model_outputs):
    return [yaml.safe_load(out.replace('`', '').replace('yaml', '')
                           .replace('`', '').replace('\\n', '\r\n')) if isinstance(out, str)
                            else out for out in model_outputs]


# add properties to dict_to_store as property:dict_property[property] if present
def add_property_if_present(dict_to_store, props, dict_property):
    for prop in props:
        if prop in dict_property:
            dict_to_store[prop] = dict_property[prop]


# configs useful in place of print results for import model
def config_to_print_import_model(configs) -> dict:
    conf_to_print = {}
    add_property_if_present(conf_to_print, [
        'name',
        'temperature',
        'tokenizer',
        'max_new_tokens',
        'do_sample',
        'top_p',
        'quantization',
    ], configs)
    return conf_to_print


# configs useful in place of print results for apis model
def config_to_print_api_model(configs) -> dict:
    conf_to_print = {}
    add_property_if_present(conf_to_print, [
        'name',
        'label',
        'deployment',
        'api_version',
        'temperature',
        'max_tokens',
        'n_responses',
        'stop',
        'top_p',
        'top_k',
    ], configs)
    return conf_to_print


# write model_output in file ex_name-model-timestamp.yml
# model_output is the list of outputs
def append_metrics(dir_label, file, metrics):
    old_yaml = load_yaml(f'{outputs}{dir_label}/{file}')
    old_yaml['metrics'] = metrics
    write_yaml(f'{outputs}{dir_label}/{file}', old_yaml)

# write model_output in file ex_name-model-timestamp.yml
# model_output is the list of outputs
def store_output(model_config, ex_config, model_output, output_preprocessed, gt_preprocessed, imported, metrics, timestamp, dir_label):
    results_output = {
        'config': config_to_print_import_model(model_config) if imported else config_to_print_api_model(model_config),
        'output': model_output,
        'output_preprocessed': output_preprocessed,
        'gt_preprocessed': gt_preprocessed,
        'metrics': metrics,
    }

    prompt_version = ex_config['prompt_version']
    ex_name = '-'.join((ex_config['name'], ex_config['version']))
    model = model_config['label'] if model_config['label'] != '' else model_config['name']

    os.makedirs(f'{outputs}{dir_label}', exist_ok=True)

    error = ""
    if metrics=={}:
        error = "-error"

    write_yaml(f'{outputs}{dir_label}/{ex_name}-{prompt_version}-{model}-{timestamp}{error}', results_output)

def get_output_file_name(ex_version, prompt_version, model):
    return f'output-{ex_version}-{prompt_version}-{model}.csv'

def get_output_file_path(ex_version, prompt_version, model, label_dir):
    f_name = f'output-{ex_version}-{prompt_version}-{model}.csv'
    return f'{auto_outputs}{label_dir}/{f_name}'

def store_automatic_output(model_config, ex_config, model_output, imported, metrics_list, timestamp, label_dir):
    for i, metrics in enumerate(metrics_list):
        data = dict()

        for key, value in ex_config.items():
            data[f"ex_{key}"] = value

        data["timestamp"] = timestamp

        data['index'] = i+1

        for key, value in config_to_print_import_model(model_config).items() if imported else config_to_print_api_model(model_config).items():
            data[f"config_{key}"] = value

        data['fact'] = model_output[i]['fact']['name']

        data['measures'] = [measure['name'] for measure in model_output[i]['measures']] if model_output[i]['measures'] else None

        data['dependencies'] = [dependency for dependency in model_output[i]['dependencies']]

        for elem in metrics:
            for met, val in metrics[elem].items():
                data[f"{elem}_{met}"] = val

        prompt_version = ex_config['prompt_version']
        model = model_config['label'] if model_config['label'] != '' else model_config['name']

        file_path = get_output_file_path(ex_config['version'], prompt_version, model, label_dir)

        write_headers = False
        headers = list(data.keys())

        try:
            # Attempt to read the first row (headers) from the CSV file
            with open(f'{file_path}', 'r+') as file:
                reader = csv.reader(file)
                existing_headers = next(reader)  # Read the first row (headers)

            # Check if the existing headers match the desired headers
            # TODO what to do in this case?
            if existing_headers != headers:
                print("Headers do not match. Writing data anyway.")
        except:
            write_headers = True

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(f'{file_path}', "a+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_headers:
                headers = list(data.keys())
                writer.writerow(headers)
            writer.writerow(list(data.values()))
