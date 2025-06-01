import torch
from dataclasses import dataclass
from transformers import HfArgumentParser, AutoModelForCausalLM
from trl import (
    DPOConfig,
    ModelConfig,
    get_peft_config,
    create_reference_model
)
from datasets import  load_dataset, load_from_disk

from src.trainer.mt_rlhf import DataCollatorForPreference, MTRLTrainer
from src.utils.loader import load_model, load_tokenizer
from src.utils.utils import setup_logging, init_wandb_training, init_seed


def build_mtrl_dataset(tokenizer, data_path, max_seq_length):
    train_dataset = load_dataset(split="train", path=data_path)
    def split_prompt_and_responses(example):
        alig_idx = example['alignment_idx'] 
        prompt = example['safe_conv'][:alig_idx*2+1]
        safe_response = example['safe_conv'][alig_idx*2+1]['content']
        bad_response = example['bad_conv'][alig_idx*2+1]['content']
        safe_reward = example["safe_reward"]
        unsafe_reward = example["unsafe_reward"]
        prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        chosen = safe_response + tokenizer.eos_token + '\n'
        rejected = bad_response + tokenizer.eos_token + '\n'
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_reward": safe_reward,
            "rejected_reward": unsafe_reward
        }
    train_dataset = train_dataset.map(split_prompt_and_responses, remove_columns=train_dataset.column_names, num_proc=8)
    return train_dataset


@dataclass
class DPOScriptArguments:
    dataset_name: str = "trl-lib/kto-mix-14k"
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    task_type: str = "helpful"
    wandb_project: str = "DPO"

def train():
    parser = HfArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    init_seed(seed=42)
    ###############
    # Setup logging
    ###############
    logger = setup_logging(script_args, training_args, model_args)
    if "wandb" in training_args.report_to:
        init_wandb_training(wandb_project=script_args.wandb_project)
    ################
    # Load tokenizer
    ################
    logger.info("*** Initializing tokenizer kwargs ***")
    tokenizer = load_tokenizer(model_args.model_name_or_path)
    ################
    # Load datasets
    ################
    logger.info("*** Initializing dataset kwargs ***")
    train_dataset = build_mtrl_dataset(tokenizer, script_args.dataset_name, training_args.max_length)
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    model = load_model(tokenizer, model_args, training_args,  AutoModelForCausalLM)
    ref_model = None
    ################
    # Training
    ################
    logger.info("*** Train ***")
    dpo_trainer = MTRLTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=training_args,
                    processing_class=tokenizer,
                    data_collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id),
                    train_dataset=train_dataset,
                    peft_config=get_peft_config(model_args)
                )

    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()


if __name__ == "__main__":
    train()

