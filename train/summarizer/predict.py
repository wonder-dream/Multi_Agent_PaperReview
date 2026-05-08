"""
摘要生成推理脚本

提供两种使用方式:
    1. 命令行: 输入论文文本，输出结构化摘要和审稿意见
    2. Python API: 供Agent层直接调用 generate_summary()

Usage:
    # 基本摘要生成 (使用预训练BART，无需微调)
    python -m train.summarizer.predict --text "We introduce a new model..."

    # 使用微调后的BART
    python -m train.summarizer.predict \
        --generative_model checkpoints/summarizer/best_model.pt \
        --text "..."

    # 配合NER/RE结果生成审稿意见
    python -m train.summarizer.predict \
        --text "..." \
        --entities_file entities.json \
        --output_file summary_output.json
"""

import os
import sys
import json
import argparse
import logging

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.summarizer.paper_summarizer import PaperSummarizer, generate_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="摘要生成推理")

    parser.add_argument("--generative_model", type=str, default=None)
    parser.add_argument("--generative_model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--entities_file", type=str, default=None)

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

    logger.info("正在加载摘要模型...")
    summarizer = PaperSummarizer(
        generative_model_path=args.generative_model,
        generative_model_name=args.generative_model_name,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        device=device
    )

    entities, triples, paper_info = [], [], {}
    if args.entities_file and os.path.exists(args.entities_file):
        with open(args.entities_file, "r", encoding="utf-8") as f:
            extraction = json.load(f)
        entities = extraction.get("entities", [])
        triples = extraction.get("triples", [])

    if args.text:
        logger.info("=" * 50)
        logger.info("单篇论文摘要生成")
        logger.info("=" * 50)

        result = generate_summary(args.text, summarizer, entities, triples, paper_info)

        logger.info(f"\n抽取式骨架 ({len(result['extractive_skeleton'])} 句):")
        for i, s in enumerate(result["extractive_skeleton"]):
            logger.info(f"  [{i+1}] {s[:100]}...")

        logger.info(f"\n抽象式摘要:\n  {result['abstractive_summary'][:200]}")

        logger.info(f"\n结构化摘要:")
        for k, v in result["structured_summary"].items():
            v_str = str(v)[:120] if v else "(空)"
            logger.info(f"  {k}: {v_str}")

        if result["review_draft"].get("overall_assessment"):
            logger.info(f"\n审稿意见:")
            review = result["review_draft"]
            logger.info(f"  优势: {review.get('strengths', [])[:2]}")
            logger.info(f"  不足: {review.get('weaknesses', [])[:2]}")
            logger.info(f"  建议: {review.get('suggestions', [])[:2]}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"\n结果已保存到 {args.output_file}")

    elif args.input_file:
        logger.info("=" * 50)
        logger.info("批量摘要生成")
        logger.info("=" * 50)

        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        texts = [item.get("text", "") for item in input_data if item.get("text")]
        logger.info(f"共 {len(texts)} 篇论文待处理")

        results = []
        for i, text in enumerate(texts):
            if (i + 1) % 5 == 0:
                logger.info(f"  进度: {i+1}/{len(texts)}")
            result = generate_summary(text, summarizer, entities, triples, paper_info)
            result["id"] = i
            results.append(result)

        output_file = args.output_file or args.input_file.replace(".json", "_summaries.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
