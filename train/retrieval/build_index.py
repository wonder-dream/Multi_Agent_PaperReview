"""
构建 FAISS 论文语义索引

从论文语料构建可供检索的语义索引:
    1. 加载论文数据 (合并数据 + SciTLDR)
    2. 用 SPECTER 编码
    3. 构建 FAISS 索引
    4. 保存索引供 Agent 层加载

Usage:
    python -m train.retrieval.build_index \
        --processed_data_dir processed_data \
        --output_dir checkpoints/retrieval

    python -m train.retrieval.build_index \
        --encoder_model allenai/specter \
        --index_type hnsw \
        --batch_size 32
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

from models.retrieval import PaperEncoder, FAISSIndex
from utils.classifier_utils import set_seed, get_device, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="构建论文语义索引")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--encoder_model", type=str, default="allenai/specter")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--index_type", type=str, default="flat_ip",
                        choices=["flat_ip", "hnsw"])
    parser.add_argument("--max_papers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="checkpoints/retrieval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_all_papers(data_dir: str, max_papers: int = 0) -> list:
    """加载所有可用论文"""
    sources = [
        os.path.join(data_dir, "arxiv_PeerRead_merge_data", "classification",
                     "merged_train.jsonl"),
        os.path.join(data_dir, "arxiv_PeerRead_merge_data", "classification",
                     "merged_dev.jsonl"),
        os.path.join(data_dir, "arxiv_PeerRead_merge_data", "classification",
                     "merged_test.jsonl"),
    ]

    papers = {}
    for src in sources:
        if not os.path.exists(src):
            continue
        logger.info(f"加载: {src}")
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                pid = item.get("arxiv_id", item.get("source", ""))
                if not text or pid in papers:
                    continue
                papers[pid] = {
                    "text": text,
                    "paper_id": pid,
                    "domain": item.get("label", ""),
                    "source": item.get("source", ""),
                    "year": item.get("year", ""),
                }

    # 同时加载 SciTLDR 数据
    sci_path = os.path.join(data_dir, "sciTLDR_data", "train.jsonl")
    if os.path.exists(sci_path):
        logger.info(f"加载 SciTLDR: {sci_path}")
        with open(sci_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                pid = item.get("paper_id", "")
                if pid in papers:
                    continue
                title = item.get("title", "")
                abstract = " ".join(item.get("target", []))
                text = f"Title: {title} Abstract: {abstract}"
                if len(text.strip()) > 20:
                    papers[pid] = {
                        "text": text,
                        "paper_id": pid,
                        "domain": "",
                        "source": "SciTLDR",
                        "year": "",
                    }

    result = list(papers.values())
    logger.info(f"去重后总计: {len(result)} 篇论文")

    if max_papers > 0 and max_papers < len(result):
        np.random.seed(42)
        indices = np.random.choice(len(result), max_papers, replace=False)
        result = [result[i] for i in indices]
        logger.info(f"限制为 {max_papers} 篇")

    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("构建论文语义索引")
    logger.info("=" * 60)

    logger.info("[1/4] 加载论文语料...")
    papers = load_all_papers(args.processed_data_dir, args.max_papers)
    texts = [p["text"] for p in papers]
    paper_ids = [p["paper_id"] for p in papers]
    logger.info(f"共 {len(papers)} 篇论文")

    logger.info(f"[2/4] 加载编码器: {args.encoder_model}")
    encoder = PaperEncoder(model_name=args.encoder_model, device=device,
                           max_length=args.max_length)
    logger.info(f"向量维度: {encoder.dim}")

    logger.info("[3/4] 编码论文...")
    start = time.time()
    embeddings = encoder.encode(texts, batch_size=args.batch_size)
    logger.info(f"编码完成: {embeddings.shape}, 用时: {format_time(time.time() - start)}")

    logger.info(f"[4/4] 构建 FAISS 索引 ({args.index_type})...")
    index = FAISSIndex(dim=encoder.dim, index_type=args.index_type)
    index.add(embeddings, paper_ids)
    index.save(args.output_dir)

    logger.info(f"\n索引构建完成! 共 {index.size} 篇论文")
    logger.info(f"索引目录: {args.output_dir}")


if __name__ == "__main__":
    main()
