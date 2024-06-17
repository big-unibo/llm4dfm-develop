from pathlib import Path

import openai
from tqdm import tqdm

from models import load_model_and_tokenizer, model_import_batch, model_api_batch, is_model_supporting_system_chat
from utils import load_yaml_conf, load_prompts, store_output, get_chat_entry, \
    load_text_and_first_prompt

model_config = load_yaml_conf(f'{Path().absolute()}/pipeline/config.yml')

if model_config['use'] == 'import':
    config = model_config['model_import']

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config['name'], config['key'], config['quantization'])

    model_outputs = []
    chat = []

    system_text = load_text_and_first_prompt(model_config['exercise']['name'], model_config['exercise'][
        'prompt_version'], config['name'])

    chat.append(get_chat_entry('system', system_text, config['name']))

    prompts = load_prompts(model_config['exercise']['prompt_version'], config['name'])[1:]

    # batch text and prompts
    with tqdm(desc=f'Prompt {config["name"]}', total=len(prompts or [])+1) as bar_batch:
        if model_config['debug_prints']:
            print(f'[pipeline] batching chat: {chat}')
        model_output = model_import_batch(model, tokenizer, chat, config, model_config['debug_prints'])
        model_outputs.append(model_output)
        chat.append(get_chat_entry('assistant', model_output, config['name']))
        bar_batch.update(1)

        for prompt in prompts:
            chat.append(get_chat_entry(prompt['role'], prompt['content'], config['name']))
            model_output = model_import_batch(model, tokenizer, chat, model_config['debug_prints'])
            model_outputs.append(model_output)
            chat.append(get_chat_entry('assistant', model_output, config['name']))
            bar_batch.update(1)

    if model_config['debug_prints']:
        print(f'[pipeline] chat: {chat}\n output: {model_output}')

    chat_input = [sentence for sentence in chat if sentence['role'] == 'system' or sentence['role'] == 'user']

    # store output
    store_output(config, model_config['exercise'], chat_input, model_outputs, False)

elif model_config['use'] == 'api':

    config = model_config['model_api']
    openai.api_key = config['key']

    model_outputs = []
    chat = []

    system_text = load_text_and_first_prompt(model_config['exercise']['name'], model_config['exercise'][
        'prompt_version'], config['name'])

    chat.append(get_chat_entry('system', system_text, config['name']))

    prompts = load_prompts(model_config['exercise']['prompt_version'], config['name'])[1:]

    # batch text and prompts
    with tqdm(desc=f'Prompt {config["name"]}', total=len(prompts or [])+1) as bar_batch:
        model_output = model_api_batch(openai, config, chat, model_config['debug_prints'])
        model_outputs.append(model_output)
        chat.append(get_chat_entry('assistant', model_output, config['name']))
        bar_batch.update(1)

        for prompt in prompts:
            chat.append(get_chat_entry(prompt['role'], prompt['content'], config['name']))
            if model_config['debug_prints']:
                print(f'[pipeline] before batching chat: {chat}')
            model_output = model_api_batch(openai, config, chat, model_config['debug_prints'])
            if model_config['debug_prints']:
                print(f'[pipeline] after batching output: {model_output}')
            model_outputs.append(model_output)
            chat.append(get_chat_entry('assistant', model_output, config['name']))
            bar_batch.update(1)

    chat_input = [sentence for sentence in chat if sentence['role'] == 'system' or sentence['role'] == 'user']

    # store output
    store_output(config, model_config['exercise'], chat_input, model_outputs, True)
