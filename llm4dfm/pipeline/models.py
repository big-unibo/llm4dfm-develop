import transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv
from typing import Callable, List
import openai
import google.generativeai as genai
import time
import requests
import json
import yaml

load_dotenv()
base_path = os.path.dirname(os.path.abspath(__file__))

save_directory = f'{base_path}/{os.getenv('SAVE_MODELS')}'

models_not_supporting_system_chat = ['mistral']
models_supporting_terminators = ['llama-3']
models_without_constraints_chat = ['gpt']


def log(message):
    print(f'{os.path.splitext(os.path.basename(__file__))[0]} - {message}\n')

# Format chat for instruct models chat template
def format_chat_for_instruct_hf_models(chat, tokenizer):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    else:
        return '\n'.join([ct['content'] for ct in chat])

def load_generate_import_function(name, model, tokenizer, config, debug_print, device, chat_template=False) -> Callable[[str], str]:
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

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": config['max_new_tokens'],
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "do_sample": config['do_sample'],
        "temperature": config['temperature'],
        "top_p": config['top_p'],
    }

    def generate_mistral_from_model(model_to_use, chat_template):

        def generate_mistral(chat):

            if debug_print:
                log(f'Batching chat: {chat}')

            formatted_chat = chat if not chat_template else format_chat_for_instruct_hf_models(chat, tokenizer)

            if debug_print:
                log(f'Batching chat formatted: {formatted_chat}')

            input_ids = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

            # Generate
            output_text = model_to_use(input_ids, **generation_kwargs)

            if debug_print:
                log(f'Decoded_batch: {output_text[0]["generated_text"]}')

            return output_text[0]["generated_text"]

        return generate_mistral

    def generate_llama_from_model(model_to_use):
        def generate_llama(chat):
            raise Exception("Not Implemented")

    def generate_llama_hf_from_model(model_to_use, chat_template):

        def generate_llama_hf(chat):

            if debug_print:
                log(f'Batching chat: {chat}')

            formatted_chat = chat if not chat_template else format_chat_for_instruct_hf_models(chat, tokenizer)

            if debug_print:
                log(f'Batching chat formatted: {formatted_chat}')

            output_text = model_to_use(
                formatted_chat,
                max_new_tokens=config['max_new_tokens'],
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=config['do_sample'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                return_full_text=False,
            )

            if debug_print:
                log(f'Decoded_batch: {output_text}')

            return output_text[0]['generated_text'] if not chat_template else output_text[0]['generated_text'].replace('assistant\n\n---\n', '').replace('assistant\n\n', '', 1)

        return generate_llama_hf

    def generate_falcon_from_model(model_to_use, chat_template):
        def generate_falcon(chat):

            if debug_print:
                log(f'Batching chat: {chat}')

            formatted_chat = chat if not chat_template else format_chat_for_instruct_hf_models(chat, tokenizer)

            if debug_print:
                log(f'Batching chat formatted: {formatted_chat}')

            output_text = model_to_use(
                formatted_chat,
                max_new_tokens=config['max_new_tokens'],
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=config['do_sample'],
                temperature=config['temperature'],
                top_p=config['top_p'],
                return_full_text=False,
            )

            if debug_print:
                log(f'Decoded_batch: {output_text}')

            return output_text[0]['generated_text'].replace('<|assistant|>', '')

        return generate_falcon

    model_to_use = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        tokenizer=tokenizer,
        device_map="auto",
    )

    match name:
        case 'mistral-7B-inst-v0.3-hf':
            return generate_mistral_from_model(model_to_use, chat_template)
        case 'llama-3-12B-inst-hf' | 'llama-3.1-8B-inst-hf' | 'llama-3.1-8B-hf' | 'llama-3.2-1B-hf' | 'llama-3.2-1B-inst-hf' | 'llama-3.2-3B-hf' | 'llama-3.2-3B-inst-hf' | 'llama-3.3-hf' | 'llama-2-7B-hf' | 'llama-2-13B-hf':
            return generate_llama_hf_from_model(model_to_use, chat_template)
        case 'falcon-3-7B-inst-hf' | 'falcon-3-10B-inst-hf' | 'falcon-3-10B-base-hf':
            return generate_falcon_from_model(model_to_use, chat_template)
        case _:
            raise Exception(f"No model generation found for {name}")

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

    def __init__(self, use, name, config, key, device, debug_print):
        self.chat = []
        self.name = name
        self.config = config
        self.config['debug_prints'] = debug_print
        if use == 'import':
            self.model, self.tokenizer = load_model_and_tokenizer(name, key, device)
            self.generate = load_generate_import_function(name, self.model, self.tokenizer, config, debug_print, device, chat_template=False)
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


# return a new chat (list of dict {'role': role, 'content': content}) entry
def get_chat_entry(entry_role, entry_content, model):
    if entry_role == 'system':
        model_name_to_use = model.split('-')[0].lower() if len(model.split('-')) > 1 else model.lower()
        if not is_model_supporting_system_chat(model_name_to_use):
            return {'role': 'user', 'content': entry_content}
    return {'role': entry_role, 'content': entry_content}


def get_chat_template(model_name, tokenizer):
    # TODO check for other models' chat template
    match model_name:
        # case 'llama-3':
        #    return {
        #         "bos_token_id": tokenizer.bos_token_id,
        #         "eos_token_id": tokenizer.eos_token_id,
        #         "pad_token_id": tokenizer.pad_token_id,
        #         "prefix": "<|startoftext|>",
        #         "suffix": "<|endoftext|>",
        #     }
        case _:
            return {}


def is_model_from_hf(model_name):
    return 'hf' in model_name


def load_model_and_tokenizer(model_name, key, device):

    match model_name:
        case 'llama-3.2-1B':
            m_name = 'Llama3.2-1B-Instruct'
        case 'llama-3.2-3B':
            m_name = 'Llama3.2-3B-Instruct'
        case 'llama-3.3':
            m_name = 'Llama3.3-70B-Instruct'
        case 'llama-3-12B-inst-hf':
            m_name = 'ehristoforu/llama-3-12b-instruct'
        case 'llama-3.1-8B-inst-hf':
            m_name = 'meta-llama/Llama-3.1-8B-Instruct'
        case 'llama-3.1-8B-hf':
            m_name = 'meta-llama/Llama-3.1-8B'
        case 'llama-3.2-1B-inst-hf':
            m_name = 'meta-llama/Llama-3.2-1B-Instruct'
        case 'llama-3.2-1B-hf':
            m_name = 'meta-llama/Llama-3.2-1B'
        case 'llama-3.2-3B-inst-hf':
            m_name = 'meta-llama/Llama-3.2-3B-Instruct'
        case 'llama-3.2-3B-hf':
            m_name = 'meta-llama/Llama-3.2-3B'
        case 'llama-3.3-hf':
            m_name = 'meta-llama/Llama-3.3-70B-Instruct'
        case 'llama-2-7B-hf':
            m_name = 'meta-llama/Llama-2-7b-chat-hf'
        case 'llama-2-13B-hf':
            m_name = 'meta-llama/Llama-2-13b-chat-hf'
        case 'falcon-3-7B-inst-hf':
            m_name = 'tiiuae/Falcon3-7B-Instruct'
        case 'falcon-3-10B-inst-hf':
            m_name = 'tiiuae/Falcon3-10B-Instruct'
        case 'falcon-3-10B-base-hf':
            m_name = 'tiiuae/Falcon3-10B-Base'
        case 'mistral-7B-inst-v0.3-hf':
            m_name = 'mistralai/Mistral-7B-Instruct-v0.3'
        case _:
            raise Exception("Model not found")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_directory = save_directory + m_name.replace(" ", "-").replace("/", "-") + '/'

    if is_model_from_hf(model_name):
        model_save_name = 'pytorch_model.bin'
        tokenizer_save_name = 'tokenizer_config.json'

        # Check if the saved model directory already contains the model files
        model_exist = os.path.exists(os.path.join(model_directory, model_save_name))
        tokenizer_exist = os.path.exists(os.path.join(model_directory, tokenizer_save_name))

        if model_exist and tokenizer_exist:
            # Load the model and tokenizer from the saved directory
            model = AutoModelForCausalLM.from_pretrained(model_directory,
                                                         torch_dtype=torch.float16 if torch.cuda.is_available() else
                                                            torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_directory)
        else:
            # Download and load the model and tokenizer from Hugging Face
            model = AutoModelForCausalLM.from_pretrained(m_name,
                                                         torch_dtype=torch.float16 if torch.cuda.is_available() else
                                                            torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(m_name, token=key)

            # Save the model and tokenizer
            model.save_pretrained(model_directory)
            tokenizer.save_pretrained(model_directory)

    else:
        raise Exception("Loading models without hf not implemented yet")

    # Move to GPU only if available
    if torch.cuda.is_available() and device == 'gpu':
        print("Cuda available, set to use GPU")
        model.to("cuda")
    else:
        if device == 'cpu':
            print("Set to use CPU")
        else:
            print("Cuda not available, set to use CPU")

    return model, tokenizer


# Check if model supports system role in chat
def is_model_supporting_system_chat(model_name):
    return model_name not in models_not_supporting_system_chat


def is_model_without_chat_constraints(model_name):
    return model_name in models_without_constraints_chat
