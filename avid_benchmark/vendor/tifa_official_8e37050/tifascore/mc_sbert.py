"""SBERT multiple-choice mapper derived from official TIFA (Apache-2.0)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SBERTModel:
    def __init__(self, ckpt="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt)
        self.model.eval()
        self._choice_embedding_cache = {}
        if torch.cuda.is_available():
            self.model.cuda()
            print("Using SBERT on GPU")

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_sentences(self, sentences):
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.model.device))
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.detach().cpu()

    def multiple_choice(self, answer, choices):
        answer_embedding = self.embed_sentences([answer])
        cache_key = tuple(choices)
        choices_embedding = self._choice_embedding_cache.get(cache_key)
        if choices_embedding is None:
            choices_embedding = self.embed_sentences(list(cache_key))
            self._choice_embedding_cache[cache_key] = choices_embedding
        top_choice_index = torch.argmax(
            torch.matmul(choices_embedding, answer_embedding.T)
        ).item()
        return choices[top_choice_index]
