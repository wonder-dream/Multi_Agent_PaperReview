"""
检索质量评估脚本

评估指标:
    - Recall@K: 前K个结果中命中相关论文的比例
    - MRR: 平均倒数排名
    - NDCG@10: 归一化折现累积增益

Usage:
    python -m train.retrieval.evaluate \
        --index_dir checkpoints/retrieval \
        --test_data processed_data/sciTLDR_data/test.jsonl \
        --output_dir results/retrieval
"""

import os
import sys
import json
import argparse
import logging
import time

import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.retrieval import PaperEncoder, FAISSIndex
from utils.classifier_utils import set_seed, get_device, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="评估检索质量")
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, default="allenai/specter")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(relevant: set, retrieved: list, k: int) -> float:
    for i, pid in enumerate(retrieved[:k]):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i, pid in enumerate(retrieved[:k]):
        rel = 1.0 if pid in relevant else 0.0
        dcg += rel / np.log2(i + 2)
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def main():
    args = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"[1/3] 加载 FAISS 索引: {args.index_dir}")
    index = FAISSIndex.load(args.index_dir)
    logger.info(f"索引包含 {index.size} 篇论文")

    logger.info(f"[2/3] 加载编码器: {args.encoder_model}")
    encoder = PaperEncoder(model_name=args.encoder_model, device=device,
                           max_length=args.max_length)

    logger.info(f"[3/3] 评估检索质量...")
    queries = []

    if args.test_data and os.path.exists(args.test_data):
        # 用 SciTLDR 作为查询
        with open(args.test_data, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                source_text = " ".join(item.get("source", []))
                title = item.get("title", "")
                query_text = f"Title: {title} Abstract: {source_text[:200]}"
                pid = item.get("paper_id", "")
                queries.append({"text": query_text, "paper_id": pid, "title": title})

    if len(queries) > args.num_queries:
        np.random.seed(42)
        queries = list(np.random.choice(queries, args.num_queries, replace=False))

    # 自评估: 随机抽取部分论文作为查询，该论文自身应被检索到
    if not queries:
        logger.info("无测试数据，使用自评估方案: 随机选择论文作为查询")
        sample_ids = np.random.choice(index.paper_ids, min(args.num_queries, index.size), replace=False)
        queries = [{"text": "", "paper_id": pid} for pid in sample_ids]
        # 用 paper_id 对应的文本（需重新编码）

    logger.info(f"共 {len(queries)} 条查询")

    total_recall = defaultdict(float)
    total_mrr = defaultdict(float)
    total_ndcg = defaultdict(float)
    eval_ks = [1, 3, 5, 10]

    for qi, query in enumerate(queries):
        if (qi + 1) % 50 == 0:
            logger.info(f"  进度: {qi+1}/{len(queries)}")

        query_emb = encoder.encode_single(query["text"])
        distances, indices = index.search(query_emb, top_k=max(eval_ks))
        retrieved_ids = [index.paper_ids[i] for i in indices[0] if i < len(index.paper_ids)]
        relevant = {query["paper_id"]}

        for k in eval_ks:
            total_recall[k] += recall_at_k(relevant, retrieved_ids, k)
            total_mrr[k] += mrr(relevant, retrieved_ids, k)
            total_ndcg[k] += ndcg_at_k(relevant, retrieved_ids, k)

    logger.info("\n" + "=" * 50)
    logger.info("检索质量评估结果")
    logger.info("=" * 50)

    n = len(queries)
    results = {}
    for k in eval_ks:
        r = total_recall[k] / n
        m = total_mrr[k] / n
        nd = total_ndcg[k] / n
        results[f"recall@{k}"] = round(r, 4)
        results[f"mrr@{k}"] = round(m, 4)
        results[f"ndcg@{k}"] = round(nd, 4)
        logger.info(f"  Recall@{k}: {r:.4f} | MRR@{k}: {m:.4f} | NDCG@{k}: {nd:.4f}")

    result_path = os.path.join(args.output_dir, "retrieval_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "num_queries": n, "index_size": index.size},
                  f, indent=2)
    logger.info(f"\n结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
