from pathlib import Path
from tqdm import tqdm
from models import Model, load_text_and_first_prompt
from utils import load_yaml_conf, load_prompts, store_output

model_config = load_yaml_conf(f'{Path().absolute()}/pipeline/config.yml')

if model_config['use'] == 'import':
    config = model_config['model_import']

    model = Model(model_config['use'], config['name'], config, config['key'],
                  model_config['debug_prints'], config['quantization'])

elif model_config['use'] == 'api':

    config = model_config['model_api']

    model = Model(model_config['use'], config['name'], config, config['key'], model_config['debug_prints'])

else:
    raise Exception("No models")

model_outputs = []
prompts = []

system_text = load_text_and_first_prompt(model_config['exercise']['name'], model_config['exercise'][
    'prompt_version'], config['name'])
prompts.append(system_text)
prompts.extend(load_prompts(model_config['exercise']['prompt_version'], config['name'])[1:])

# batch text and prompts
with (tqdm(desc=f'Prompt {config["name"]}', total=len(prompts)) as bar_batch):
    for prompt in prompts:
        model_output = model.batch(prompt)
        model_outputs.append(model_output)
        bar_batch.update(1)

if model_config['debug_prints']:
    print(f'[pipeline] chat: {model.chat}\n output: {model_output}')

chat_input = [sentence for sentence in model.chat if sentence['role'] == 'system' or sentence['role'] == 'user']

# store output
store_output(config, model_config['exercise'], chat_input, model_outputs, model_config['use'] == 'import')
