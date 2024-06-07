from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import bfloat16
import os
from dotenv import load_dotenv

load_dotenv()

save_directory = os.getenv('SAVE_MODELS')


def load_model_and_tokenizer(model_name, key, quantization):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    ) if quantization else ''
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
            m_name = ''

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
                                                     # quantization_config=bnb_config,
                                                     # device_map='auto',
                                                     token=key
                                                     )
        tokenizer = AutoTokenizer.from_pretrained(m_name,
                                                  token=key
                                                  )

        # Save the model and tokenizer
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)

    return model, tokenizer


# TODO text is a chat now
# compute a batch given a model, a tokenizer and input_text, returning results
def model_import_batch(model, tokenizer, text) -> str:
    encoded = tokenizer.apply_chat_template(text, tokenize=False, return_tensors="pt")

    generated_ids = model.generate(**encoded, max_new_tokens=1000, do_sample=True)
    decoded_with_decode = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    decoded_with_batch = tokenizer.batch_decode(generated_ids)

    print(f'[models] -> decoded_decode: {decoded_with_decode}')
    print(f'[models] -> decoded_batch: {decoded_with_batch}')

    # with torch.no_grad():
    #     model_outputs = model.generate(**text_to_use)

    # generated_text = tokenizer.decode(model_outputs[0], skip_special_tokens=True)

    return decoded_with_decode


# TODO text is a chat now
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
