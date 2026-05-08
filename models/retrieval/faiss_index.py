"""
FAISS 向量索引管理器

功能:
    - 构建论文语义索引 (Flat IP / HNSW)
    - 加载/保存索引
    - 相似论文检索
    - 相似度批量计算

索引类型:
    - Flat IP: 精确内积搜索，适合小规模 (<100K)
    - HNSW: 近似最近邻，适合大规模 (快速但略有精度损失)
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import faiss


class FAISSIndex:
    """FAISS 论文语义索引"""

    def __init__(self, dim: int, index_type: str = "flat_ip"):
        """
        Args:
            dim: 向量维度
            index_type: "flat_ip" (精确内积) 或 "hnsw" (近似搜索)
        """
        self.dim = dim
        self.index_type = index_type
        self.index = None
        self.paper_ids = []
        self._build_index()

    def _build_index(self):
        if self.index_type == "flat_ip":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        print(f"[FAISSIndex] 索引类型: {self.index_type}, 维度: {self.dim}")

    def add(self, embeddings: np.ndarray, paper_ids: List[str] = None):
        """添加向量到索引"""
        embeddings = embeddings.astype(np.float32)
        if paper_ids:
            self.paper_ids.extend(paper_ids)
        else:
            start = len(self.paper_ids)
            self.paper_ids.extend([f"paper_{i}" for i in range(start, start + len(embeddings))])
        self.index.add(embeddings)
        print(f"[FAISSIndex] 已添加 {len(embeddings)} 条, 总计: {self.index.ntotal}")

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的 top_k 篇论文

        Args:
            query_emb: (1, dim) 或 (dim,) 查询向量
            top_k: 返回最相似的数量

        Returns:
            (distances, indices) - 距离和内积索引
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb.astype(np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_emb, k)
        return distances, indices

    def get_papers(self, indices: np.ndarray, distances: np.ndarray = None) -> List[Dict]:
        """根据索引返回论文ID和分数"""
        results = []
        if distances is not None and len(distances) > 0:
            scores = distances[0] if distances.ndim == 2 else distances
        else:
            scores = [0.0] * len(indices[0]) if indices.ndim == 2 else [0.0] * len(indices)
        for i, score in zip(indices[0] if indices.ndim == 2 else indices, scores):
            i = int(i)
            if i < len(self.paper_ids):
                results.append({"paper_id": self.paper_ids[i], "score": float(score)})
        return results

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """计算两个向量的余弦相似度 (假设已L2归一化)"""
        emb1 = emb1.astype(np.float32).reshape(1, -1)
        emb2 = emb2.astype(np.float32).reshape(1, -1)
        return float(np.dot(emb1, emb2.T)[0, 0])

    def save(self, save_dir: str):
        """保存索引和论文ID列表"""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "papers.index"))
        with open(os.path.join(save_dir, "paper_ids.json"), "w", encoding="utf-8") as f:
            json.dump({"paper_ids": self.paper_ids, "dim": self.dim,
                        "index_type": self.index_type, "count": self.index.ntotal}, f)
        print(f"[FAISSIndex] 已保存 {self.index.ntotal} 条索引到 {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "FAISSIndex":
        """加载索引"""
        meta_path = os.path.join(save_dir, "paper_ids.json")
        index_path = os.path.join(save_dir, "papers.index")
        if not os.path.exists(meta_path) or not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {save_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(dim=meta["dim"], index_type=meta.get("index_type", "flat_ip"))
        instance.index = faiss.read_index(index_path)
        instance.paper_ids = meta["paper_ids"]
        print(f"[FAISSIndex] 已加载 {instance.index.ntotal} 条索引 from {save_dir}")
        return instance

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
