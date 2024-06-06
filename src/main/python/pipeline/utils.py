import os
from dotenv import load_dotenv
import yaml
from datetime import datetime

load_dotenv()

datasets = os.getenv('DATASETS')
outputs = os.getenv('OUTPUTS')
results = os.getenv('RESULTS')
inputs = os.getenv('INPUTS')


# datetime object containing current date and time
def get_timestamp():
    return datetime.now().strftime('%Y/%m/%d-%H:%M:%S')


# return yaml configurations as dict
def load_yaml_conf(yaml_file) -> dict:
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


# return the text of exercise given ex_name
def load_text_exercise(ex_name):
    with open(f'{datasets}{ex_name}-text.yml', 'r') as file:
        ex_text = yaml.safe_load(file)
    return ex_text


# return prompts of exercise as a dict given ex_name and model_name
def load_prompts(version, model_name):
    with open(f'{inputs}prompts-{version}.yml', 'r') as file:
        ex_prompts = yaml.safe_load(file)
    return ex_prompts[model_name]


# given ex_text and a list of prompts, return an ordered list with ex_text first
def concat_input_text_and_prompts(text, prompts) -> list:
    return list((text, *prompts))


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
    ], configs)
    return conf_to_print


# configs useful in place of print results for apis model
def config_to_print_api_model(configs) -> dict:
    conf_to_print = {}
    add_property_if_present(conf_to_print, [
        'name',
        'temperature',
        'tokenizer',
        'max_tokens',
        'role',
        'n_responses',
        'stop'
    ], configs)
    return conf_to_print


# write model_output in file ex_name-model-timestamp.yml
# output is a list of outputs
def store_output(model, imported, configurations, model_output, ex_name):
    results_output = {
        'config': config_to_print_import_model(configurations[model]) if imported else config_to_print_api_model(configurations[model]),
        'output': model_output
    }

    prompt_version = configurations['exercise']['prompt_version']

    with open(f'{outputs}{ex_name}-{prompt_version}-{model}-{get_timestamp()}.yml', 'w') as outfile:
        yaml.dump(results_output, outfile, default_flow_style=False, sort_keys=False)
