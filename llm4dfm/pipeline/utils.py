import os
from copy import deepcopy
import pandas as pd
from dotenv import load_dotenv
import yaml
from datetime import datetime
import re
import csv

load_dotenv()

base_path = os.path.dirname(os.path.abspath(__file__))

datasets = f'{base_path}/{os.getenv("DATASETS_PATH", os.getenv("DATASETS"))}'
outputs = f'{base_path}/{os.getenv("OUTPUTS_PATH", os.getenv("OUTPUTS"))}'
results = f'{base_path}/{os.getenv("RESULTS_PATH", os.getenv("RESULTS"))}'
inputs = f'{base_path}/{os.getenv("INPUTS_PATH", os.getenv("INPUTS"))}'
auto_outputs = f'{base_path}/{os.getenv("AUTO_OUTPUTS_PATH", os.getenv("AUTO_OUTPUTS"))}'
resources = f'{base_path}/../resources/'

# General utils

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

# Standard names utils

def get_dir_label_name(ex_version, prompt_version, model_label, dir_label):
    return f"{ex_version}-{prompt_version}-{model_label}-{dir_label}"

# Extract exercise number based on last digit of exercise name
def extract_ex_num(ex_name):
    numbers = re.findall(r'\d+', ex_name)  # Find all sequences of digits
    if numbers:
        return int(numbers[-1])
    else:
        # print("No exercise numbers in exercise name.")
        return None

# Yaml utils

# return yaml configurations as dict
def load_yaml(yaml_file) -> dict:
    with open(f'{yaml_file}.yml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# return yaml configurations as dict
def load_yaml_from_resources(yaml_filename) -> dict:
    return load_yaml(f'{resources}{yaml_filename}')

def write_yaml(yaml_file, data):
    with open(f'{yaml_file}.yml', 'w+', encoding='utf-8') as file:
        yaml.dump(data, file)

def load_full_path_csv(path):
    return pd.read_csv(path)

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

def load_prompts_as_single(prompt_version, model_name, exercise):
    prompts = load_yaml(f'{inputs}prompts-{prompt_version}')
    if model_name in prompts:
        prompts = prompts[model_name]
    else:
        prompts = prompts['base']

    prompts[-1]['content'] = "\n".join([prompts[-1]['content'], load_text_exercise(exercise)])

    return prompts

# return prompts of exercise as a dict given ex_name and model_name
def load_prompts_as_multiple(version, model_name, exercise):
    prompts = load_yaml(f'{inputs}prompts-{version}')
    if model_name in prompts:
        prompts = prompts[model_name]
    else:
        prompts = prompts['base']

    # Return a list of lists to bind prompts as complete chat prompts -> first one is a prompt of dict until
    # receive a user-roled prompt, then each assistant prompt is a prompt itself
    ret_prompts = [[]]

    for idx, prompt in enumerate(prompts):
        # If still not loaded a user prompt, build up just first prompt
        if not any(sub_chat["role"] == "user" for sub_chat in ret_prompts[0]):
            if prompt['role'] == 'user':
                prompt['content'] = "\n".join([prompt['content'], load_text_exercise(exercise)])
            ret_prompts[-1].append(prompt)
        # If already loaded, each new prompt is a valid prompt
        else:
            ret_prompts.append([prompt])

    return ret_prompts

# load output exercise used in second-step and its filename (used after to store the image)
def load_output_exercise(dir_name, full_name):
    return load_yaml(f'{outputs}{dir_name}/{full_name}')

def label_edges(out, gt, tp_idx, fp_idx, fn_idx, gt_used):
    out_to_return = deepcopy(out)
    gt_to_return = deepcopy(gt)
    for idx, dep in enumerate(out_to_return['dependencies']):
        if idx in tp_idx:
            dep['label'] = 'tp'
        elif idx in fp_idx:
            dep['label'] = 'fp'
        else:
            dep['label'] = 'error'
    for idx, dep in enumerate(gt_to_return['dependencies']):
        if idx in gt_used:
            dep['label'] = 'tp'
        elif idx in fn_idx:
            dep['label'] = 'fn'
        else:
            dep['label'] = 'error'
    return out_to_return, gt_to_return


# Map output to a valid yaml as dict
def output_as_valid_yaml(model_output):

    if not isinstance(model_output, str):
        return model_output

    # Match YAML inside triple backticks
    match = re.search(r"```(?:yaml|yml)?\s*(.*?)```", model_output, re.DOTALL)

    if match:
        # Extract content inside triple backticks, removing `yaml` or `yml` if present
        yaml_content = match.group(1)
    else:
        match_preprocess = re.search(r"```(?:yaml|yml)?\s*(.*?)```", model_output + '```', re.DOTALL)

        if match_preprocess:
            yaml_content = match_preprocess.group(1)
        else:
            yaml_content = model_output

    return yaml.safe_load(yaml_content.replace('`', '').replace('yaml', '')
                             .replace('yml', '').replace('\\n', '\r\n'))

# Store output utils

# add properties to dict_to_store as property:dict_property[property] if present
def add_property_if_present(dict_to_store, props, dict_property):
    for prop in props:
        if prop in dict_property:
            dict_to_store[prop] = dict_property[prop]


def config_to_print(configs) -> dict:
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
def store_output(model_config, prompt_version, ex_name, ex_version, model_output, output_preprocessed, gt_preprocessed, imported, metrics, error_detection, timestamp, dir_label):
    results_output = {
        'config': config_to_print(model_config),
        'output': model_output,
        'output_preprocessed': output_preprocessed,
        'gt_preprocessed': gt_preprocessed,
        'metrics': metrics,
        'errors': error_detection,
    }

    ex_name_to_use = '-'.join((ex_name, ex_version))
    model = model_config['label'] if model_config['label'] != '' else model_config['name']

    os.makedirs(f'{outputs}{dir_label}', exist_ok=True)

    error = ""
    if metrics=={}:
        error = "-error"

    write_yaml(f'{outputs}{dir_label}/{ex_name_to_use}-{prompt_version}-{model}-{timestamp}{error}', results_output)


# write model_output in file ex_name-model-timestamp.yml
def store_additional_properties(dir_label, ex_name, props):
    prev_out = load_yaml(f'{outputs}{dir_label}/{ex_name}')

    for prop in props:
        prev_out[prop] = props[prop]
    write_yaml(f'{outputs}{dir_label}/{ex_name}', prev_out)


def get_csv_file_from_output_dir(dir_name):
    directory = f'{outputs}{dir_name}'
    file = f'output-{dir_name}.csv'

    return os.path.join(directory, file)


def get_headers_csv():
    return ['ex_name','ex_version','ex_prompt_version','ex_number','timestamp','index','time','config_name','config_label',
            'config_deployment','config_api_version','config_temperature','config_max_tokens','config_n_responses',
            'config_stop','config_top_p','config_top_k','fact','measures','dependencies','node_tp','node_fp','node_fn',
            'edges_tp','edges_fn','edges_fp','edges_precision','edges_recall','edges_f1','nodes_tp','nodes_fn',
            'nodes_fp','nodes_precision','nodes_recall','nodes_f1','errors_dependencies_reversed',
            'errors_dependencies_missing','errors_dependencies_extra','errors_measures_missing',
            'errors_measures_extra','errors_fact_incorrect','errors_fact_false_fact','errors_attributes_shared_missing',
            'errors_attributes_shared_extra','errors_attributes_shared_with_fact_root_missing',
            'errors_attributes_shared_with_fact_root_extra','errors_miscellaneous_extra_disconnected_components',
            'errors_miscellaneous_extra_tags']


def store_csv(model_config, ex_name, ex_version, ex_prompt_version, ex_num, output_preprocessed, imported, metrics_list, detected_list, timestamp, label_dir, times):

    for i, out_prep in enumerate(output_preprocessed):
        data = dict()

        data["ex_name"] = ex_name
        data["ex_version"] = ex_version
        data["ex_prompt_version"] = ex_prompt_version
        data["ex_number"] = ex_num

        data["timestamp"] = timestamp

        data['index'] = i+1

        data['time'] = f'{times[i]:.4f}'

        for key, value in config_to_print(model_config).items():
            data[f"config_{key}"] = value

        try:
            data['fact'] = out_prep['fact']['name']
        except:
            print(f'[Utils] Error loading {i}-th output preprocessed, len out_prep = {len(output_preprocessed)}, len metrics_list = {len(metrics_list)}\n\nOutput preprocessed: {output_preprocessed}\n\nMetrics: {metrics_list}')
            raise Exception('Error storing csv')

        data['measures'] = [measure['name'] for measure in output_preprocessed[i]['measures']] if output_preprocessed[i]['measures'] else None

        data['dependencies'] = [dependency for dependency in output_preprocessed[i]['dependencies']]

        for node in out_prep['nodes']:
            data[f'node_{node}'] = out_prep['nodes'][node]

        for elem in metrics_list[i]:
            for met, val in metrics_list[i][elem].items():
                data[f"{elem}_{met}"] = val

        for det, val in detected_list[i].items():
            if isinstance(val, dict):
                for sub_det, prop in val.items():
                    data[f'errors_{det}_{sub_det}'] = prop
            else:
                data[f'errors_{det}'] = val

        file_path = get_csv_file_from_output_dir(label_dir)

        write_headers = False
        headers = get_headers_csv()

        for head in headers:
            if head not in data:
                data[head] = None

        try:
            # Attempt to read the first row (headers) from the CSV file
            with open(f'{file_path}', 'r+') as file:
                reader = csv.reader(file)
                existing_headers = next(reader)  # Read the first row (headers)

            if existing_headers != headers:
                print(f"Headers do not match\nActual:{existing_headers}\nNew: {headers}\nUpdating headers.")

                # Load existing
                df = pd.read_csv(file_path)

                for head in headers:
                    if head not in df.columns:
                        df[head] = None
                df.to_csv(file_path, index=False)
        except:
            write_headers = True

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(f'{file_path}', "a+", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if write_headers:
                writer.writerow(headers)
            writer.writerow([data[key] for key in headers])


def update_csv(dir_name, timestamp, ex_name, ex_num, ex_version, output_preprocessed, metrics_list, detected):
    path = get_csv_file_from_output_dir(dir_name)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=get_headers_csv())

    for idx, out_prep in enumerate(output_preprocessed):
        if not df.empty:
            missing_headers = [col for col in get_headers_csv() if col not in df.columns]

            # Add missing headers with default NaN values
            for col in missing_headers:
                df[col] = pd.NA
            matching_rows = (df['ex_number'] == ex_num) & (df['timestamp'] == timestamp) & (df['index'] == idx+1)
        else:
            matching_rows = None

        data = dict()

        data['fact'] = out_prep['fact']['name']

        data['measures'] = [measure['name'] for measure in out_prep['measures']] if out_prep['measures'] else None

        data['dependencies'] = [dependency for dependency in out_prep['dependencies']]

        for node in out_prep['nodes']:
            data[f'node_{node}'] = out_prep['nodes'][node]

        for det, val in detected[idx].items():
            if isinstance(val, dict):
                for sub_det, prop in val.items():
                    data[f'errors_{det}_{sub_det}'] = prop
            else:
                data[f'errors_{det}'] = val

        for elem in metrics_list[idx]:
            for met, val in metrics_list[idx][elem].items():
                data[f"{elem}_{met}"] = val

        if not df.empty and (not matching_rows is None) and matching_rows.any():
            for prop in data:
                df.loc[matching_rows, prop] = str(data[prop])
        else:
            data['ex_name'] = ex_name
            data['ex_version'] = ex_version
            data['ex_number'] = ex_num
            data['timestamp'] = timestamp
            data['index'] = idx + 1

            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(path, index=False)
