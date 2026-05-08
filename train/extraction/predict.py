"""
信息抽取推理脚本

提供两种使用方式:
    1. 命令行: 输入文本或JSON文件，输出实体和关系
    2. Python API: 供Agent层直接调用 extract_information()

Usage:
    # 单条文本抽取
    python -m train.extraction.predict \
        --ner_model checkpoints/ner/best_model.pt \
        --re_model checkpoints/re/best_model.pt \
        --text "BERT achieves state-of-the-art results on SQuAD with F1 score of 93.2."

    # 批量抽取 (JSON文件)
    python -m train.extraction.predict \
        --ner_model checkpoints/ner/best_model.pt \
        --re_model checkpoints/re/best_model.pt \
        --input_file papers.json \
        --output_file extraction_results.json
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.extraction.paper_extractor import PaperExtractor, extract_information

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="信息抽取推理")

    parser.add_argument("--ner_model", type=str, required=True)
    parser.add_argument("--re_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)

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

    logger.info(f"加载NER模型: {args.ner_model}")
    logger.info(f"加载RE模型: {args.re_model}")

    extractor = PaperExtractor(
        ner_model_path=args.ner_model,
        re_model_path=args.re_model,
        model_name=args.model_name,
        max_length=args.max_length,
        device=device
    )

    if args.text:
        logger.info("=" * 50)
        logger.info("单条文本信息抽取")
        logger.info("=" * 50)

        result = extract_information(args.text, extractor)

        logger.info(f"\n输入文本: {args.text[:120]}...")
        logger.info(f"\n抽取结果:")
        logger.info(f"  实体数: {len(result['entities'])}")
        for e in result["entities"]:
            logger.info(f"    [{e['type']}] {e['text']}")
        logger.info(f"  三元组数: {len(result['triples'])}")
        for t in result["triples"]:
            logger.info(f"    {t['head']} --[{t['relation']}]--> {t['tail']}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"\n结果已保存到 {args.output_file}")
        else:
            logger.info("\n完整结果:\n" + json.dumps(result, indent=2, ensure_ascii=False))

    elif args.input_file:
        logger.info("=" * 50)
        logger.info("批量信息抽取")
        logger.info("=" * 50)

        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        if isinstance(input_data, list):
            texts = [item.get("text", "") for item in input_data if item.get("text")]
        elif isinstance(input_data, dict):
            texts = [input_data.get("text", "")]
        else:
            logger.error("输入文件格式错误")
            return

        logger.info(f"共 {len(texts)} 条文本待抽取")

        results = []
        for i, text in enumerate(texts):
            if (i + 1) % 10 == 0:
                logger.info(f"  处理进度: {i+1}/{len(texts)}")
            result = extract_information(text, extractor)
            result["id"] = i
            results.append(result)

        total_entities = sum(len(r["entities"]) for r in results)
        total_triples = sum(len(r["triples"]) for r in results)
        logger.info(f"\n抽取完成: {total_entities} 个实体, {total_triples} 个三元组")

        entity_types = {}
        relation_types = {}
        for r in results:
            for e in r["entities"]:
                entity_types[e["type"]] = entity_types.get(e["type"], 0) + 1
            for t in r["triples"]:
                relation_types[t["relation"]] = relation_types.get(t["relation"], 0) + 1

        logger.info("\n实体类型分布:")
        for t, c in sorted(entity_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {t}: {c}")

        logger.info("\n关系类型分布:")
        for t, c in sorted(relation_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {t}: {c}")

        output_file = args.output_file or args.input_file.replace(".json", "_extraction.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
