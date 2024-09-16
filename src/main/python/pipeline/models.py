from transformers import pipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from torch import bfloat16
# import torch
import os
from dotenv import load_dotenv
from typing import Callable, List
import openai
import google.generativeai as genai
import time
from utils import load_text_exercise, load_prompts
import requests
import json
import yaml

load_dotenv()

save_directory = os.getenv('SAVE_MODELS')

models_not_supporting_system_chat = ['mistral']
models_supporting_terminators = ['llama-3']
models_without_constraints_chat = ['gpt']


def log(message):
    print(f'{os.path.splitext(os.path.basename(__file__))[0]} - {message}\n')


# def load_generate_import_function(name, model, tokenizer, config, debug_print) -> Callable[[str], str]:
#     # Default values
#     pad_token_id = None
#     eos_token_id = None
#
#     # Check if the model supports eos_token_id
#     if config['name'] in models_supporting_terminators:
#         eos_token_id = [
#             tokenizer.eos_token_id,
#             tokenizer.convert_tokens_to_ids("<|eot_id|>")
#         ]
#     else:
#         pad_token_id = tokenizer.eos_token_id
#
#     def generate_with_import(chat):
#         if debug_print:
#             log(f'Batching chat: {chat}')
#         encoded = tokenizer.apply_chat_template(chat, return_tensors="pt")
#
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 encoded,
#                 max_new_tokens=config['max_new_tokens'],
#                 eos_token_id=eos_token_id,
#                 pad_token_id=pad_token_id,
#                 do_sample=config['do_sample'],
#                 temperature=config['temperature'],
#                 top_p=config['top_p'],
#             )
#         decoded_with_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#
#         if debug_print:
#             log(f'Decoded_batch: {decoded_with_batch}')
#
#         return decoded_with_batch
#
#     match name:
#         case 'falcon':
#             return generate_with_import
#         case _:
#             return generate_with_import


def load_generate_api_function(name, model, config, debug_print) -> Callable[[List[str]], str]:
    def generate_with_gtp_api(chat):
        if debug_print:
            log(f'Batching chat: {chat}')

        endpoint = os.getenv(f'ENDPOINT_{name.upper()}')
        api_version = config['api_version']
        deployment_name = config['deployment']

        headers = {
            "Content-Type": "application/json",
            "api-key": config['key']
        }

        data = {
            'messages': chat,
            'max_tokens': config['max_tokens'],
            'n': config['n_responses'],
            'stop': config['stop'],
            'temperature': config['temperature'],
            'top_p': config['top_p'],
        }

        response = requests.post(
            f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}",
            headers=headers,
            data=json.dumps(data)
        )
        if response.status_code == 200:
            # Parse and print the response
            result = response.json()
            if debug_print:
                log(f'Output_message: {result}')
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Gemini send prompt through chat, parent function's signature model is the chat required by the model to prompt
    def generate_with_gemini_api(chat):
        if debug_print:
            log(f'Batching chat: {chat}')

        text_response = []

        responses = (model.send_message(
            chat[-1]['content'],  # TODO check if only last sentence should be passed
            # it doesn't support role chat
            generation_config=genai.types.GenerationConfig(
                candidate_count=config['n_responses'],
                stop_sequences=config['stop'],
                max_output_tokens=config['max_tokens'],
                top_p=config['top_p'],
                top_k=config['top_k'],
                temperature=config['temperature'],
            )
        ))

        for chunk in responses:
            text_response.append(chunk.text)

        output_message = "".join(text_response)

        if debug_print:
            log(f'Output_message: {output_message}')

        return output_message

    def generate_with_huggingface_api(chat):
        if debug_print:
            log(f'Batching chat: {chat}')

        generation_config = {
            "max_length": config['max_tokens'],
            "num_return_sequences": config['n_responses'],
            "temperature": config['temperature'],
            "top_k": config['top_k'],
            "top_p": config['top_p'],
        }

        generated_text = model(chat[-1], **generation_config)

        if debug_print:
            log(f'Output_message: {generated_text}')

        return generated_text

    match name:
        case 'gpt':
            return generate_with_gtp_api
        case 'gemini':
            return generate_with_gemini_api
        case 'mistral':
            return generate_with_huggingface_api
        case _:
            raise Exception("Generate function for this model not implemented yet")


class Model:

    def __init__(self, use, name, config, key, debug_print, quantization=False):
        self.chat = []
        self.name = name
        self.config = config
        self.config['debug_prints'] = debug_print
        if use == 'import':
            print('import not supported')
            # self.model, self.tokenizer = load_model_and_tokenizer(name, key, quantization)
            # self.generate = load_generate_import_function(name, self.model, self.tokenizer, config, debug_print)
        elif use == 'api':
            self.model = load_model_api(name, key)
            if 'gemini' in self.config['name']:
                self.model = self.model.start_chat(history=[])
            self.generate = load_generate_api_function(name, self.model, config, debug_print)

    def batch(self, prompt):
        batched = False
        wait = 3
        max_tries = 5
        m_output = ''
        # Prompt can be list if it's the first input
        if type(prompt) is list:
            for p in prompt:
                self.chat.append(get_chat_entry(p['role'], p['content'], self.name))
        else:
            self.chat.append(get_chat_entry(prompt['role'], prompt['content'], self.name))
        while not batched and max_tries>0:
            try:
                m_output = self.generate(self.chat)
                batched = True
            except Exception as e:
                print(f'Model batch error [{e}] trying in {wait} seconds')
                time.sleep(wait)
                wait += 1
                max_tries -= 1
        try:
            m_output = yaml.safe_load(m_output)
        except yaml.YAMLError as _:
            pass
        self.chat.append(get_chat_entry('assistant', m_output, self.name))
        return m_output

    def refresh_session(self):
        self.chat = []


# Load first prompt as system, providing scenario
# Then load text exercise and second prompt as user
# Return as list
def load_text_and_first_prompt(ex_name, version, model_name):
    ex_text = load_text_exercise(ex_name)
    prompts = load_prompts(version, model_name)
    scenario_prompt = prompts[0]
    if len(prompts) > 1:
        second_prompt = prompts[1]
        return [scenario_prompt, get_chat_entry(second_prompt['role'], '\n'.join([second_prompt['content'], ex_text]),
                                            model_name)]
    else:
        return [scenario_prompt, get_chat_entry('user', ex_text, model_name)]


# return a new chat (list of dict {'role': role, 'content': content}) entry
def get_chat_entry(entry_role, entry_content, model):
    if entry_role == 'system':
        if not is_model_supporting_system_chat(model):
            return {'role': 'user', 'content': entry_content}
    return {'role': entry_role, 'content': entry_content}


# def get_chat_template(model_name, tokenizer):
#     # TODO check for other models' chat template
#     match model_name:
#         case 'llama-3':
#             return {
#                 "bos_token_id": tokenizer.bos_token_id,
#                 "eos_token_id": tokenizer.eos_token_id,
#                 "pad_token_id": tokenizer.pad_token_id,
#                 "prefix": "<|startoftext|>",
#                 "suffix": "<|endoftext|>",
#             }
#         case _:
#             return {}


# def load_model_and_tokenizer(model_name, key, quantization):
#     # TODO work on quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,  # 4-bit quantization
#         bnb_4bit_quant_type='nf4',  # Normalized float 4
#         bnb_4bit_use_double_quant=True,  # Second quantization after the first
#         bnb_4bit_compute_dtype=bfloat16  # Computation type
#     ) if quantization else None
#     match model_name:
#         case 'llama-3':
#             m_name = 'meta-llama/Meta-Llama-3-8B'
#         case 'llama-2':
#             m_name = 'meta-llama/Llama-2-7b-hf'
#         case 'falcon':
#             m_name = 'tiiuae/falcon-11B'
#         case 'mistral':
#             m_name = 'mistralai/Mistral-7B-Instruct-v0.1'
#         case _:
#             raise Exception("Model not found")
#
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
#
#     model_directory = save_directory + m_name.replace(" ", "-").replace("/", "-") + '/'
#     model_save_name = 'pytorch_model.bin'
#     tokenizer_save_name = 'tokenizer_config.json'
#
#     # Check if the saved model directory already contains the model files
#     model_files_exist = os.path.exists(os.path.join(model_directory, model_save_name))
#     tokenizer_files_exist = os.path.exists(os.path.join(model_directory, tokenizer_save_name))
#
#     if model_files_exist and tokenizer_files_exist:
#         # Load the model and tokenizer from the saved directory
#         model = AutoModelForCausalLM.from_pretrained(model_directory)
#         tokenizer = AutoTokenizer.from_pretrained(model_directory)
#     else:
#         # Download and load the model and tokenizer from Hugging Face
#         model = AutoModelForCausalLM.from_pretrained(m_name,
#                                                      # trust_remote_code=True,
#                                                      quantization_config=bnb_config,
#                                                      # device_map='auto',
#                                                      token=key,
#                                                      )
#         tokenizer = AutoTokenizer.from_pretrained(m_name,
#                                                   token=key
#                                                   )
#
#         # Save the model and tokenizer
#         model.save_pretrained(model_directory)
#         tokenizer.save_pretrained(model_directory)
#
#     chat_template = get_chat_template(model_name, tokenizer)
#     if chat_template:
#         tokenizer.add_special_tokens(chat_template)
#         model.resize_token_embeddings(len(tokenizer))
#
#     return model, tokenizer


def load_model_api(name, key):
    match name:
        case 'gpt':
            openai.api_key = key
            return openai
        case 'gemini':
            genai.configure(api_key=key)
            return genai.GenerativeModel('gemini-1.5-flash')
        case 'mistral':
            pipe = pipeline("conversational",
                            model="mistralai/Mistral-7B-v0.3",
                            use_auth_token=True)  # To use it, it's required huggingface cli login
            return pipe
        case _:
            raise Exception("Model not found")


# Check if model supports system role in chat
def is_model_supporting_system_chat(model_name):
    return model_name not in models_not_supporting_system_chat


def is_model_without_chat_constraints(model_name):
    return model_name in models_without_constraints_chat
