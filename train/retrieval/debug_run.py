"""
检索 Debug 运行脚本 - 快速验证语义检索引擎

步骤:
    1. 加载论文语料 (合并数据)
    2. 编码为向量 (SPECTER)
    3. 构建 FAISS 索引
    4. 执行检索查询验证

Usage:
    python -m train.retrieval.debug_run --sample_size 500 --encoder_model allenai/specter
    python -m train.retrieval.debug_run --sample_size 200 --encoder_model allenai/scibert_scivocab_uncased
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.retrieval import PaperEncoder, FAISSIndex, PaperRetriever
from utils.classifier_utils import set_seed, get_device, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug运行 (语义检索)")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--encoder_model", type=str, default="allenai/specter")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="checkpoints/debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def load_papers(data_dir: str, sample_size: int, seed: int = 42) -> list:
    """从合并数据加载论文"""
    train_file = os.path.join(data_dir, "arxiv_PeerRead_merge_data",
                              "classification", "merged_train.jsonl")
    papers = []
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if not text:
                    continue
                paper = {
                    "text": text,
                    "paper_id": item.get("arxiv_id", item.get("source", str(len(papers)))),
                    "domain": item.get("label", ""),
                    "source": item.get("source", ""),
                    "year": item.get("year", ""),
                }
                papers.append(paper)

    logger.info(f"已加载 {len(papers)} 篇论文")
    if sample_size < len(papers):
        np.random.seed(seed)
        indices = np.random.choice(len(papers), sample_size, replace=False)
        papers = [papers[i] for i in indices]
        logger.info(f"采样 {sample_size} 篇")
    return papers


def main():
    args = parse_args()

    if args.run_name is None:
        from datetime import datetime
        ts = datetime.now().strftime("%m%d_%H%M")
        model_tag = args.encoder_model.replace("/", "_")
        args.run_name = f"debug_retrieval_{model_tag}_{ts}"

    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info(f"Debug 运行 - 语义检索")
    logger.info(f"编码器: {args.encoder_model} | 采样: {args.sample_size}")
    logger.info("=" * 60)

    logger.info("[1/5] 加载论文语料...")
    papers = load_papers(args.processed_data_dir, args.sample_size, args.seed)

    logger.info(f"[2/5] 加载编码器 ({args.encoder_model})...")
    encoder = PaperEncoder(model_name=args.encoder_model, device=device,
                           max_length=args.max_length)
    logger.info(f"向量维度: {encoder.dim}")

    logger.info(f"[3/5] 编码论文...")
    start = time.time()
    texts = [p["text"] for p in papers]
    embeddings = encoder.encode(texts, batch_size=args.batch_size)
    logger.info(f"编码完成: {embeddings.shape}, 用时: {format_time(time.time() - start)}")

    logger.info(f"[4/5] 构建 FAISS 索引...")
    index = FAISSIndex(dim=encoder.dim)
    paper_ids = [p["paper_id"] for p in papers]
    index.add(embeddings, paper_ids)
    index.save(args.output_dir)
    logger.info(f"索引大小: {index.size} 篇")

    logger.info(f"[5/5] 检索验证...")
    query_texts = [
        "Title: BERT: Pre-training of Deep Bidirectional Transformers. Abstract: We introduce a new language representation model called BERT.",
        "Title: ImageNet Classification with Deep Convolutional Networks. Abstract: We trained a large, deep convolutional neural network.",
        "Title: Attention Is All You Need. Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
    ]

    for i, query in enumerate(query_texts):
        logger.info(f"\n--- 查询 {i+1} ---")
        logger.info(f"Query: {query[:100]}...")

        distances, indices = index.search(encoder.encode_single(query), top_k=5)
        papers_found = index.get_papers(indices, distances)

        if papers_found:
            for rank, p in enumerate(papers_found):
                matched = next((x for x in papers if x["paper_id"] == p["paper_id"]), {})
                text_preview = matched.get("text", "")[:80] if matched else "N/A"
                logger.info(f"  #{rank+1}: {p['paper_id']} (score={p['score']:.4f}) | {text_preview}...")
        else:
            logger.info("  (无结果)")

    logger.info(f"\n索引已保存: {args.output_dir}")
    logger.info("一切正常!")


if __name__ == "__main__":
    main()
