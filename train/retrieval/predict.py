"""
语义检索推理脚本

提供三种检索模式:
    1. 相似论文检索 (semantic_search)
    2. 重复性检测 (detect_similarity)
    3. 论文对相似度计算 (pairwise_similarity)

Agent调用接口:
    >>> from models.retrieval import PaperRetriever, semantic_search, detect_similarity
    >>> retriever = PaperRetriever(index_dir="checkpoints/retrieval")
    >>> results = semantic_search("Title: BERT... Abstract: ...", retriever, top_k=5)

Usage:
    python -m train.retrieval.predict \
        --index_dir checkpoints/retrieval \
        --text "Title: BERT: Pre-training... Abstract: We introduce..."

    python -m train.retrieval.predict \
        --index_dir checkpoints/retrieval \
        --input_file queries.json \
        --output_file results.json
"""

import os
import sys
import json
import argparse
import logging

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.retrieval import PaperRetriever, semantic_search, detect_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="语义检索推理")
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--encoder_model", type=str, default="allenai/specter")
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text2", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input_file is None:
        logger.error("请提供 --text 或 --input_file 参数")
        return

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"加载语义检索引擎: {args.index_dir}")
    retriever = PaperRetriever(
        encoder_model=args.encoder_model,
        index_dir=args.index_dir,
        device=device
    )

    if args.text and args.text2:
        # 两篇论文相似度计算
        sim = retriever.pairwise_similarity(args.text, args.text2)
        logger.info("=" * 50)
        logger.info(f"论文语义相似度: {sim:.4f}")
        if sim >= args.threshold:
            logger.info(f"判定: 高度相似 (>= {args.threshold})")
        else:
            logger.info(f"判定: 正常差异 (< {args.threshold})")

    elif args.text:
        # 单篇论文检索
        logger.info("=" * 50)
        logger.info("相似论文检索")
        logger.info("=" * 50)
        logger.info(f"Query: {args.text[:150]}...")

        results = semantic_search(args.text, retriever, top_k=args.top_k)

        logger.info(f"\n检索结果 (Top-{len(results)}):")
        for rank, r in enumerate(results):
            logger.info(f"  #{rank+1}: {r['paper_id']} (score={r['score']:.4f})")

        # 重复性检测
        similar = detect_similarity(args.text, retriever, threshold=args.threshold)
        if similar:
            logger.info(f"\n潜在重复/高度相似论文 ({len(similar)} 篇, 阈值>{args.threshold}):")
            for s in similar:
                logger.info(f"  {s['paper_id']} (score={s['score']:.4f})")
        else:
            logger.info(f"\n未发现高度相似论文 (阈值>{args.threshold})")

        if args.output_file:
            result = {"query": args.text[:200], "results": results, "similar": similar}
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"\n结果已保存到 {args.output_file}")

    elif args.input_file:
        # 批量检索
        logger.info("=" * 50)
        logger.info("批量语义检索")
        logger.info("=" * 50)

        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        queries = [item.get("text", "") for item in input_data if item.get("text")]

        all_results = []
        for i, query in enumerate(queries):
            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i+1}/{len(queries)}")
            results = semantic_search(query, retriever, top_k=args.top_k)
            all_results.append({"id": i, "query": query[:200], "results": results})

        output_file = args.output_file or args.input_file.replace(".json", "_retrieval.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
