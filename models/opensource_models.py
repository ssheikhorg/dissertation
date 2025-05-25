from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf_model(model_name):
    model_map = {
        'llama2': 'meta-llama/Llama-2-7b-chat-hf',
        'mistral': 'mistralai/Mistral-7B-v0.1'
    }

    tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
    model = AutoModelForCausalLM.from_pretrained(model_map[model_name])
    return model, tokenizer
