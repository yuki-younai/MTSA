from typing import List, Callable, Union
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sentence_transformers import SentenceTransformer, util

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class CosineSentenceEmbeddingReward(object):
    def __init__(self, model_path, n_samples: int = -1, gpu = 0):
        self.device = gpu
        self.n_samples = n_samples

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

        self.sentence_embeddings = None
        self.sentence_embeddings_valid = None
        
    def compute_similarity(self, X: List[str], Y: List[str]) -> torch.Tensor:
        phi_X = self._compute_embeddings(X)
        phi_Y = self._compute_embeddings(Y)
        sim = torch.diagonal(util.pytorch_cos_sim(phi_X, phi_Y), 0)
        return sim

    def compute_l1_div_rewards(self, X: List[str]) -> np.ndarray:
        LAM_ADV = 0.5        
        LAM_DIV1 = 100
        LAM_DIV2 = 5    
        embeddings = self._compute_embeddings(X).detach().cpu().numpy()
        dist_matrix = torch.tensor(manhattan_distances(embeddings)) / LAM_DIV1
        div_reward = -1 * torch.mean(torch.exp(-dist_matrix), dim=1) * LAM_DIV2
        return div_reward.cpu().numpy()

    def _compute_embeddings(self, sentences: List[str]) -> torch.Tensor:
        # sentence_embeddings = None  # Initialize to None
        with torch.no_grad():
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def append_reference(self, ref: Union[str, List[str]]):
        if not isinstance(ref, List):
            ref = [ref]
        # Keep embeddings on the same device as model to avoid device mismatch
        sentence_embeddings = self._compute_embeddings(ref)
        if self.sentence_embeddings is None:
            self.sentence_embeddings = sentence_embeddings
        else:
            self.sentence_embeddings = torch.cat([self.sentence_embeddings, sentence_embeddings], dim=0)

    def __call__(self, hypotheses: List[str]) -> np.ndarray:
        if self.sentence_embeddings is None or len(self.sentence_embeddings) == 0:
            return np.zeros(len(hypotheses))
        hypo_sentence_embeddings = self._compute_embeddings(hypotheses)
        
        # Ensure ref_sentence_embeddings is on the same device as hypo_sentence_embeddings before similarity computation
        if self.n_samples > 0:
            sample_size = min(len(self.sentence_embeddings), self.n_samples)
            sample_indices = random.sample(range(self.sentence_embeddings.shape[0]), k=sample_size)
            ref_sentence_embeddings = self.sentence_embeddings[sample_indices, :]
        else:
            ref_sentence_embeddings = self.sentence_embeddings
        
        # Move ref_sentence_embeddings to the same device as hypo_sentence_embeddings if necessary
        ref_sentence_embeddings = ref_sentence_embeddings.to(hypo_sentence_embeddings.device)

        sims = pairwise_cosine_similarity(hypo_sentence_embeddings, ref_sentence_embeddings)
        return sims.mean(dim=1).cpu().numpy()