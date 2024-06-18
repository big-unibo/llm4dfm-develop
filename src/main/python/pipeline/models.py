from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import bfloat16
import torch
import os
from dotenv import load_dotenv
from typing import Callable, List
import openai

load_dotenv()

save_directory = os.getenv('SAVE_MODELS')

models_not_supporting_system_chat = ['mistral']
models_supporting_terminators = ['llama-3']


def load_generate_import_function(name, model, tokenizer, config, debug_print) -> Callable[[str], str]:

    # Default values
    pad_token_id = None
    eos_token_id = None

    # Check if the model supports eos_token_id
    if config['name'] in models_supporting_terminators:
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        pad_token_id = tokenizer.eos_token_id

    def generate_with_import(chat):
        if debug_print:
            print(f'[models] batching chat: {chat}')
        encoded = tokenizer.apply_chat_template(chat, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                encoded,
                max_new_tokens=config['max_new_tokens'],
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=config['do_sample'],
                temperature=config['temperature'],
                top_p=config['top_p'],
            )
        decoded_with_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        if debug_print:
            print(f'[models] -> decoded_batch: {decoded_with_batch}')

        return decoded_with_batch

    match name:
        case 'falcon':
            return generate_with_import
        case _:
            return generate_with_import


def load_generate_api_function(name, model, config, debug_print) -> Callable[[List[str]], str]:
    def generate_with_gtp_api(chat):
        if debug_print:
            print(f'[models] batching chat: {chat}')
        response = model.ChatCompletion.create(
            model=config['name'],
            messages=chat,
            max_tokens=config['max_tokens'],
            n=config['n_responses'],
            stop=config['stop'],
            temperature=config['temperature'],
        )
        output_message = response['choices'][0]['message']['content']  # TODO works only with gpt ?
        if debug_print:
            print(f'[models] -> output_message: {output_message}')
        return output_message

    match name:
        case 'gpt':
            return generate_with_gtp_api
        case _:
            return lambda chat: chat


class Model:

    def __init__(self, use, name, config, key, debug_print, quantization=False):
        self.chat = []
        self.name = name
        self.config = config
        self.config['debug_prints'] = debug_print
        if use == 'import':
            self.model, self.tokenizer = load_model_and_tokenizer(name, key, quantization)
            self.generate = load_generate_import_function(name, self.model, self.tokenizer, config, debug_print)
        elif use == 'api':
            self.model = load_model_api(name, key)
            self.generate = load_generate_api_function(name, self.model, config, debug_print)

    def batch(self, prompt):
        if self.config['debug_prints']:
            print(f'[models] -> batching: {prompt}')
        self.chat.append(get_chat_entry(prompt.role, prompt.content, self.name))
        model_output = self.generate(self.chat)
        self.chat.append(get_chat_entry('assistant', model_output, self.name))
        return model_output

    def refresh_session(self):
        self.chat = []


# return a new chat (list of dict {'role': role, 'content': content}) entry
def get_chat_entry(entry_role, entry_content, model):
    if entry_role == 'system':
        if not is_model_supporting_system_chat(model):
            return {'role': 'user', 'content': entry_content}
    return {'role': entry_role, 'content': entry_content}


def get_chat_template(model_name, tokenizer):
    # TODO check for other models' chat template
    match model_name:
        case 'llama-3':
            return {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "prefix": "<|startoftext|>",
                "suffix": "<|endoftext|>",
            }
        case _:
            return {}


def load_model_and_tokenizer(model_name, key, quantization):
    # TODO work on quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    ) if quantization else None
    match model_name:
        case 'llama-3':
            m_name = 'meta-llama/Meta-Llama-3-8B'
        case 'llama-2':
            m_name = 'meta-llama/Llama-2-7b-hf'
        case 'falcon':
            m_name = 'tiiuae/falcon-11B'
        case 'mistral':
            m_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        case _:
            raise Exception("Model not found")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_directory = save_directory + m_name.replace(" ", "-").replace("/", "-") + '/'
    model_save_name = 'pytorch_model.bin'
    tokenizer_save_name = 'tokenizer_config.json'

    # Check if the saved model directory already contains the model files
    model_files_exist = os.path.exists(os.path.join(model_directory, model_save_name))
    tokenizer_files_exist = os.path.exists(os.path.join(model_directory, tokenizer_save_name))

    if model_files_exist and tokenizer_files_exist:
        # Load the model and tokenizer from the saved directory
        model = AutoModelForCausalLM.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
    else:
        # Download and load the model and tokenizer from Hugging Face
        model = AutoModelForCausalLM.from_pretrained(m_name,
                                                     # trust_remote_code=True,
                                                     quantization_config=bnb_config,
                                                     # device_map='auto',
                                                     token=key,
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(m_name,
                                                  token=key
                                                  )

        # Save the model and tokenizer
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)

    chat_template = get_chat_template(model_name, tokenizer)
    if chat_template:
        tokenizer.add_special_tokens(chat_template)
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_model_api(name, key):
    if name == 'gpt':
        openai.api_key = key
        return openai


# use to build a working chat
def is_model_supporting_system_chat(model_name):
    return model_name not in models_not_supporting_system_chat
