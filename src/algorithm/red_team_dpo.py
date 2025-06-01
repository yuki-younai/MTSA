from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, HfArgumentParser
from trl import ModelConfig, ScriptArguments, TrlParser, create_reference_model, get_peft_config, DPOConfig, DPOTrainer
from transformers import HfArgumentParser, AutoModelForCausalLM

from torch.utils.data import Dataset, DataLoader
from datasets import  load_dataset, load_from_disk

from src.utils.loader import load_model, load_tokenizer
from src.utils.utils import setup_logging, init_wandb_training

@dataclass
class DPOScriptArguments:
    dataset_name: str = "trl-lib/kto-mix-14k"
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    task_type: str = "helpful"
    wandb_project: str = "DPO"


def build_redteam_dpo_dataset(tokenizer, data_path, max_seq_length, task_type="helpful") -> Dataset:

    dataset = load_dataset(split="train", path=data_path)

    def split_prompt_and_responses_redteam(example):
        alig_idx = example['alignment_idx'] 
        prompt = example["pos_red_conv"][:alig_idx*2+2]
        prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        pos_response = example["pos_red_conv"][alig_idx*2+2]['content'] + tokenizer.eos_token + '\n'
        neg_response = example["neg_red_conv"][alig_idx*2+2]['content'] + tokenizer.eos_token + '\n'
        
        return {
            "prompt": prompt,
            "chosen": pos_response,
            "rejected": neg_response,
        }

    return dataset.map(split_prompt_and_responses_redteam, num_proc=8)
    
def train():
    parser = HfArgumentParser((DPOScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    set_seed(seed=42)
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
    train_dataset = build_redteam_dpo_dataset(tokenizer, script_args.dataset_name, training_args.max_length)
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
    dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=training_args,
                    processing_class=tokenizer,
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
