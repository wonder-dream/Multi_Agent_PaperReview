"""
论文检索引擎封装类 (供Agent层直接调用)

提供:
    - semantic_search(): 相似论文检索
    - detect_similarity(): 重复性检测
    - recommend_citations(): 引用推荐
"""
import numpy as np
from typing import List, Dict, Optional

from .encoder import PaperEncoder
from .faiss_index import FAISSIndex


class PaperRetriever:
    """论文语义检索引擎"""

    def __init__(
        self,
        encoder_model: str = "allenai/specter",
        index_dir: Optional[str] = None,
        device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    ):
        self.device = device

        self.encoder = PaperEncoder(model_name=encoder_model, device=device)

        if index_dir:
            self.index = FAISSIndex.load(index_dir)
            print(f"[PaperRetriever] 索引已加载: {self.index.size} 篇论文")
        else:
            self.index = FAISSIndex(dim=self.encoder.dim)
            print(f"[PaperRetriever] 创建空索引 (dim={self.encoder.dim})")

    def build_index(self, papers: List[Dict], text_key: str = "text",
                    id_key: str = "paper_id", batch_size: int = 32):
        """从论文列表构建索引"""
        texts = [p.get(text_key, "") for p in papers]
        ids = [p.get(id_key, str(i)) for i, p in enumerate(papers)]
        embeddings = self.encoder.encode(texts, batch_size=batch_size)
        self.index.add(embeddings, ids)

    def semantic_search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Agent调用接口: 相似论文检索

        Args:
            query_text: 查询论文的 title + abstract
            top_k: 返回最相似的论文数量

        Returns:
            [{"paper_id": "...", "score": 0.95}, ...]
        """
        if self.index.size == 0:
            return []
        query_emb = self.encoder.encode_single(query_text)
        distances, indices = self.index.search(query_emb, top_k=top_k)
        return self.index.get_papers(indices, distances)

    def detect_similarity(self, query_text: str, threshold: float = 0.85) -> List[Dict]:
        """
        Agent调用接口: 重复性检测

        Args:
            query_text: 待检测论文文本
            threshold: 相似度阈值 (高于此值标记为潜在重复)

        Returns:
            高于阈值的相似论文列表
        """
        results = self.semantic_search(query_text, top_k=20)
        return [r for r in results if r["score"] >= threshold]

    def pairwise_similarity(self, text1: str, text2: str) -> float:
        """计算两篇论文的语义相似度"""
        emb1 = self.encoder.encode_single(text1)
        emb2 = self.encoder.encode_single(text2)
        return float(np.dot(emb1, emb2))

    def recommend_citations(self, query_text: str, candidate_ids: List[str] = None,
                            top_k: int = 5) -> List[Dict]:
        """
        Agent调用接口: 引用推荐

        检索与查询论文语义相关但未在candidate_ids中出现的论文

        Args:
            query_text: 查询论文文本
            candidate_ids: 已引用论文ID (排除)
            top_k: 推荐数量

        Returns:
            推荐引用的论文列表
        """
        results = self.semantic_search(query_text, top_k=top_k * 3)
        candidate_set = set(candidate_ids or [])
        filtered = [r for r in results if r["paper_id"] not in candidate_set]
        return filtered[:top_k]


def semantic_search(query_text: str, retriever: "PaperRetriever",
                    top_k: int = 5) -> List[Dict]:
    """Agent调用接口"""
    return retriever.semantic_search(query_text, top_k=top_k)


def detect_similarity(query_text: str, retriever: "PaperRetriever",
                      threshold: float = 0.85) -> List[Dict]:
    """Agent调用接口"""
    return retriever.detect_similarity(query_text, threshold=threshold)
