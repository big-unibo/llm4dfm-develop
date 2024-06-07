from tqdm import tqdm
import openai
from models import load_model_and_tokenizer, model_import_batch, model_api_batch
from utils import (load_yaml_conf, load_prompts, load_text_exercise, store_output, get_chat_entry)
from pathlib import Path

model_config = load_yaml_conf(f'{Path().absolute()}/pipeline/config.yml')

if model_config['use'] == 'import':
    config = model_config['model_import']

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['name'], config['key'], config['quantization'])

    # load text exercise and prompts
    exercise_text = load_text_exercise(model_config['exercise']['name'])
    prompts = load_prompts(model_config['exercise']['prompt_version'], config['name'])
    system_text = prompts['system'] if 'system' in prompts else ''
    prompts_text = prompts['chat']

    model_outputs = []
    chat = []

    for sys_text in system_text:
        chat.append(get_chat_entry('system', sys_text))

    # batch text and prompts
    with tqdm(desc=f'Prompt {config["name"]}', total=len(prompts_text)) as bar_batch:
        for idx, input_text in prompts_text:
            chat.append(get_chat_entry('user', input_text))
            model_output = model_import_batch(model, tokenizer, chat)
            model_outputs.append(model_output)
            chat.append(get_chat_entry('assistant', model_output))
            bar_batch.update(1)

    # store output
    store_output(config['name'], False, model_config, model_outputs, model_config['exercise']['name'])

elif model_config['use'] == 'api':

    config = model_config['model_api']

    # load text exercise and prompts
    exercise_text = load_text_exercise(model_config['exercise']['name'])
    prompts_text = load_prompts(model_config['exercise']['name'], config['name'])
    model_outputs = []

    # TODO check if it works with a chat or just plain text
    # concat text exercise and prompts
    inputs_list = [] # concat_input_text_and_prompts(exercise_text, prompts_text)

    openai.api_key = config['key']

    # batch text and prompts
    with tqdm(desc=f'Prompt {config["name"]}', total=len(inputs_list)) as bar_batch:
        for i in inputs_list:
            model_outputs.append(model_api_batch(openai, config, i))
            bar_batch.update(1)

    # store output
    store_output(config['name'], True, model_config, model_outputs, model_config['exercise']['name'])
