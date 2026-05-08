"""
论文语义编码器: SPECTER / SciBERT

SPECTER 是专为科学文献设计的预训练编码器，基于 citation 关系训练，
生成的向量天然适合论文相似度计算和引用推荐。

支持:
    - SPECTER (推荐): allenai/specter, allenai-specter
    - SciBERT: allenai/scibert_scivocab_uncased (CLS vector)
"""
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer


class PaperEncoder:
    """论文语义编码器"""

    def __init__(
        self,
        model_name: str = "allenai/specter",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        pooling: str = "cls"
    ):
        self.device = device
        self.max_length = max_length
        self.pooling = pooling

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            hidden = self.model.config.hidden_size
            self.dim = hidden
            print(f"[PaperEncoder] 已加载 {model_name} (dim={self.dim})")
        except Exception as e:
            print(f"[PaperEncoder] {model_name} 加载失败: {e}")
            fallback = "allenai/scibert_scivocab_uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback)
            self.model = AutoModel.from_pretrained(fallback).to(device)
            self.model.eval()
            self.dim = self.model.config.hidden_size
            print(f"[PaperEncoder] 回退到 {fallback} (dim={self.dim})")

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32,
               normalize: bool = True) -> np.ndarray:
        """
        将论文文本编码为语义向量

        Args:
            texts: 论文文本列表 (Title + Abstract)
            batch_size: 批大小
            normalize: 是否L2归一化

        Returns:
            (N, dim) 嵌入矩阵
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoding = self.tokenizer(
                batch, max_length=self.max_length, padding=True,
                truncation=True, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if self.pooling == "cls":
                vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                # Mean pooling
                mask = attention_mask.unsqueeze(-1).float().cpu().numpy()
                hidden = outputs.last_hidden_state.cpu().numpy()
                vec = (hidden * mask).sum(axis=1) / (mask.sum(axis=1) + 1e-9)

            if normalize:
                vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
            embeddings.append(vec)

        return np.concatenate(embeddings, axis=0)

    @torch.no_grad()
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """编码单篇论文"""
        return self.encode([text], normalize=normalize)[0]


def encode_papers(texts: List[str], encoder: PaperEncoder) -> np.ndarray:
    """便捷接口"""
    return encoder.encode(texts)
