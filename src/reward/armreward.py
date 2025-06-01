import argparse
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List


class ArmoRMPipeline:
    def __init__(self, model_name_or_path, device="1", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score