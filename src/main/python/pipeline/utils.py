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
def load_prompts(ex_name, model_name):
    with open(f'{inputs}prompts-v{ex_name}.yml', 'r') as file:
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
    add_property_if_present(conf_to_print, ['name',
                                            'temperature',
                                            'tokenizer'
                                            ], configs)
    return conf_to_print


# configs useful in place of print results for apis model
def config_to_print_api_model(configs) -> dict:
    conf_to_print = {}
    add_property_if_present(conf_to_print, ['name',
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
        'config': config_to_print_import_model(configurations) if imported else config_to_print_api_model(configurations),
        'output': model_output
    }

    with open(f'{outputs}{ex_name}-{model}-{get_timestamp()}.yml', 'w') as outfile:
        yaml.dump(results_output, outfile, default_flow_style=False, sort_keys=False)


# compute a batch given a model, a tokenizer and input_text, returning results
def model_import_batch(model, tokenizer, text) -> str:
    encoded = tokenizer.apply_chat_template(text, return_tensors="pt")

    generated_ids = model.generate(encoded, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    # with torch.no_grad():
    #     model_outputs = model.generate(**text_to_use)

    # generated_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

    return decoded


# compute a batch given apis and configurations
def model_api_batch(openai, config, text) -> str:
    response = openai.ChatCompletion.create(
        model=config['name'],
        messages=[
            {'role': config['role'], 'content': text},  # TODO check if role should be inserted in input
        ],
        max_tokens=config['max_tokens'],
        n=config['n_responses'],
        stop=config['stop'],
        temperature=config['temperature'],
    )

    output_message = response['choices'][0]['message']['content']  # TODO works only with gpt ?

    return output_message
