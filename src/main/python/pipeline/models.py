from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import bfloat16


def load_model_and_tokenizer(model_name, key, quantization):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    ) if quantization else ''
    match model_name:
        case 'llama-3':
            m_name = 'meta-llama/Meta-Llama-3-8B'  # meta-llama/Llama-2-7b-chat-hf
        case 'falcon':
            m_name = 'tiiuae/falcon-11B'
        case 'mistral':
            m_name = 'mistralai/Mistral-7B-Instruct-v0.1',
        case _:
            m_name = ''
    return (AutoModelForCausalLM.from_pretrained(m_name,
                                                 trust_remote_code=True,
                                                 quantization_config=bnb_config,
                                                 device_map='auto',
                                                 token=key),
            AutoTokenizer.from_pretrained(m_name, token=key))
