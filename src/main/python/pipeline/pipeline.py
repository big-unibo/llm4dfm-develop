from tqdm import tqdm
import openai
from models import load_model_and_tokenizer, model_import_batch, model_api_batch
from utils import (load_yaml_conf, load_prompts, load_text_exercise, store_output, concat_input_text_and_prompts)
from pathlib import Path

model_config = load_yaml_conf(f'{Path().absolute()}/pipeline/config.yml')

if model_config['use'] == 'import':
    config = model_config['model_import']

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['name'], config['key'], config['quantization'])

    # load text exercise and prompts
    exercise_text = load_text_exercise(model_config['exercise']['name'])
    prompts_text = load_prompts(model_config['exercise']['name'], config['name'])
    # concat text exercise and prompts
    inputs_list = concat_input_text_and_prompts(exercise_text, prompts_text)

    model_outputs = []

    # batch text and prompts
    for i in tqdm(inputs_list, desc=f'Prompt {config["name"]}'):
        model_outputs.append(model_import_batch(model, tokenizer, inputs_list[i]))

    # store output
    store_output(config['name'], False, config, model_outputs, model_config['exercise']['name'])

elif model_config['use'] == 'api':

    config = model_config['model_api']

    # load text exercise and prompts
    exercise_text = load_text_exercise(model_config['exercise']['name'])
    prompts_text = load_prompts(model_config['exercise']['name'], config['name'])
    model_outputs = []

    # concat text exercise and prompts
    inputs_list = concat_input_text_and_prompts(exercise_text, prompts_text)

    openai.api_key = config['key']

    # batch text and prompts
    for i in tqdm(inputs_list, desc=f'Prompt {config["name"]}'):
        model_outputs.append(model_api_batch(openai, config, inputs_list[i]))

    # store output
    store_output(config['name'], True, config, model_outputs, model_config['exercise']['name'])
