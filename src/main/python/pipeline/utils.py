import os
from dotenv import load_dotenv
import yaml
from datetime import datetime
import re

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')


def log(message):
    print(f'{os.path.splitext(os.path.basename(__file__))[0]} - {message}\n')


# datetime object containing current date and time
def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S')


# return yaml configurations as dict
def load_yaml(yaml_file) -> dict:
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


# return the text of exercise given ex_name
def load_text_exercise(ex_name):
    with open(f'{datasets}{ex_name}-text.yml', 'r', encoding='utf-8') as file:
        ex_text = yaml.safe_load(file)
    return ex_text['text']


# return the ground truth of exercise given ex_name
def load_ground_truth_exercise(ex_name, full_name=''):
    if full_name:
        file_name = '-'.join(full_name.split('-')[:2])
    else:
        file_name = ex_name
    with open(f'{datasets}{file_name}-ground-truth.yml', 'r', encoding='utf-8') as file:
        ex_ground_truth = yaml.safe_load(file)
    return ex_ground_truth


# load output exercise used in second-step and its filename (used after to store the image)
def load_output_exercise_and_name(ex_name, version, prompt_version, model_name, model_version, latest=True, timestamp='',
                         full_name=''):
    if full_name:
        exercise = full_name + '.yml'
    else:
        # List all files in the directory
        files = os.listdir(outputs)

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

    with open(f'{outputs}{exercise}', 'r', encoding='utf-8') as file:
        ex_output = yaml.safe_load(file)
    return ex_output, exercise


# return prompts of exercise as a dict given ex_name and model_name
def load_prompts(version, model_name):
    with open(f'{inputs}prompts-{version}.yml', 'r', encoding='utf-8') as file:
        ex_prompts = yaml.safe_load(file)
    return ex_prompts[model_name]


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
        'version',
        'api_version'
        'temperature',
        'tokenizer',
        'max_tokens',
        'n_responses',
        'stop',
        'top_p',
        'top_k',
    ], configs)
    return conf_to_print


# write model_output in file ex_name-model-timestamp.yml
# model_output is the list of outputs
def store_output(model_config, ex_config, model_output, imported, metrics):
    results_output = {
        'config': config_to_print_import_model(model_config) if imported else config_to_print_api_model(model_config),
        'output': model_output,
        'metrics': metrics,
    }

    prompt_version = ex_config['prompt_version']
    ex_name = '-'.join((ex_config['name'], ex_config['version']))
    model = model_config['label'] if model_config['label'] != '' else model_config['name']

    with open(f'{outputs}{ex_name}-{prompt_version}-{model}-{get_timestamp()}.yml', 'w+', encoding='utf-8') as outfile:
        yaml.dump(results_output, outfile, default_flow_style=False, sort_keys=False, allow_unicode=True)
