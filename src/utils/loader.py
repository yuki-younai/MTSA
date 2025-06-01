from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from trl import (
    get_quantization_config,
    get_kbit_device_map
)


def load_tokenizer(model_name_or_path: "ModelArguments", chat_template=None) -> "TokenizerModule":
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_eos_token= True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    
    return tokenizer 

def load_model(
    tokenizer, model_config, training_args, model_class
) -> "PreTrainedModel":

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        pretrained_model_name_or_path = model_config.model_name_or_path,
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = model_class.from_pretrained(**model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model










