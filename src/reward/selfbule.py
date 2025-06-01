from typing import List, Callable, Union
import os
import random
import numpy as np
from multiprocessing import Pool
import nltk
from fast_bleu import BLEU
    
class SelfBleuReward(object):
    def __init__(self, 
                 grams: List[int] = [2, 3, 4, 5], 
                 sample_size: int = -1,
                 tokenizer: Callable = nltk.word_tokenize,) -> None:

        self.references = []
        self.grams = grams
        self.sample_size = sample_size
        self.tokenizer = tokenizer
        self.references_valid = []

    def append_reference(self, ref: Union[str, List[str]]):
        if isinstance(ref, list):
            self.references += list(map(self.tokenizer, ref))
        else:
            self.references.append(self.tokenizer(ref))

    def __call__(self, hypotheses: List[str]):
        weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        
        avg_scores = []
        scores_per_ref = []
        
        for ref in self.references:
            bleu = BLEU([ref], weights)
            tokenized_hypotheses = list(map(self.tokenizer, hypotheses))
            score = bleu.get_score(tokenized_hypotheses)
            scores_per_ref.append(list(score.values()))
        
        avg_scores = np.mean(scores_per_ref, axis=0)
        
        return np.mean(avg_scores, axis=0)

    def append_reference_valid(self, ref: Union[str, List[str]]):
        if isinstance(ref, list):
            self.references_valid += list(map(self.tokenizer, ref))
        else:
            self.references_valid.append(self.tokenizer(ref))

    def valid_bleu(self, hypotheses: List[str]):

        weights = {f"{n}-gram": ([1. / n] * n) for n in self.grams}
        
        avg_scores = []
        scores_per_ref = []
        
        for ref in self.references_valid:
            bleu = BLEU([ref], weights)
            tokenized_hypotheses = list(map(self.tokenizer, hypotheses))
            score = bleu.get_score(tokenized_hypotheses)
            scores_per_ref.append(list(score.values()))
        
        avg_scores = np.mean(scores_per_ref, axis=0)
        
        return np.mean(avg_scores, axis=0)