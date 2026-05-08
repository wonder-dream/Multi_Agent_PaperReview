"""
抽取式摘要器: TextRank + SciBERT 句子相似度

基于 TextRank 图算法 + SciBERT 语义编码抽取关键句作为摘要骨架。
无需训练，开箱即用。
"""
import re
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoModel, AutoTokenizer


class ExtractiveSummarizer:
    """TextRank + SciBERT 抽取式摘要器"""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        print(f"[ExtractiveSummarizer] 已加载 {model_name}")

    def summarize(self, text: str, top_k: int = 5, diversity_lambda: float = 0.3) -> List[str]:
        """抽取 top_k 个关键句"""
        sentences = self._split_sentences(text)
        if len(sentences) <= top_k:
            return sentences

        embeddings = self._encode_sentences(sentences)
        scores = self._textrank(embeddings, diversity_lambda)
        ranked = sorted(zip(scores, sentences), key=lambda x: -x[0])
        return [s for _, s in ranked[:top_k]]

    @torch.no_grad()
    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        embeddings = []
        for sent in sentences:
            encoding = self.tokenizer(sent, max_length=256, padding="max_length",
                                      truncation=True, return_tensors="pt")
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_vec)
        return np.concatenate(embeddings, axis=0)

    def _textrank(self, embeddings: np.ndarray, diversity_lambda: float = 0.3) -> np.ndarray:
        """TextRank with MMR diversity penalty"""
        n = len(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        embeddings = embeddings / norms
        sim_matrix = np.dot(embeddings, embeddings.T)

        scores = np.ones(n) / n
        for _ in range(50):
            new_scores = np.zeros(n)
            for i in range(n):
                new_scores[i] = (1 - diversity_lambda) * np.dot(sim_matrix[i], scores)
            new_scores /= (np.sum(new_scores) + 1e-9)
            if np.max(np.abs(new_scores - scores)) < 1e-6:
                break
            scores = new_scores
        return scores

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.split()) >= 3]


def extract_key_sentences(text: str, summarizer: ExtractiveSummarizer,
                          top_k: int = 5) -> List[str]:
    """便捷接口"""
    return summarizer.summarize(text, top_k=top_k)
