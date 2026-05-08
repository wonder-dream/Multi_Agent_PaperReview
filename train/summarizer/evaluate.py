"""
摘要生成评估脚本

加载训练好的 BART 摘要模型，在测试集上评估:
    - ROUGE-1/2/L
    - BERTScore (可选)
    - 生成样例展示

Usage:
    python -m train.summarizer.evaluate \
        --model_path checkpoints/summarizer/best_model.pt \
        --test_data processed_data/sciTLDR_data/test.jsonl \
        --output_dir results/summarizer
"""

import os
import sys
import json
import argparse
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rouge import Rouge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.summarizer.generative import BARTSummarizer
from models.summarizer.dataset import SciTLDRDataset
from utils.classifier_utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
rouge = Rouge()


def parse_args():
    parser = argparse.ArgumentParser(description="评估摘要模型")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model = BARTSummarizer(model_name=args.model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"加载测试数据: {args.test_data}")
    test_dataset = SciTLDRDataset(args.test_data, tokenizer_name=args.model_name,
                                   max_source_length=args.max_source_length,
                                   max_target_length=args.max_target_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=lambda x: {
                                  k: torch.stack([d[k] for d in x]) for k in x[0]})
    logger.info(f"测试集: {len(test_dataset)} 条")

    all_preds, all_refs = [], []
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    max_length=args.max_target_length, num_beams=4)
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)

    scores = rouge.get_scores(all_preds, all_refs, avg=True)
    logger.info("\n" + "=" * 50)
    logger.info("摘要评估结果")
    logger.info("=" * 50)
    logger.info(f"ROUGE-1: {scores['rouge-1']['f']:.4f}")
    logger.info(f"ROUGE-2: {scores['rouge-2']['f']:.4f}")
    logger.info(f"ROUGE-L: {scores['rouge-l']['f']:.4f}")

    logger.info(f"\n--- 生成样例 (前 {args.num_samples} 条) ---")
    for i in range(min(args.num_samples, len(all_preds))):
        logger.info(f"\n样例 {i+1}:")
        logger.info(f"  参考: {all_refs[i][:150]}...")
        logger.info(f"  生成: {all_preds[i][:150]}...")

    result_path = os.path.join(args.output_dir, "summarizer_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"rouge-1": scores["rouge-1"]["f"], "rouge-2": scores["rouge-2"]["f"],
                   "rouge-l": scores["rouge-l"]["f"]}, f, indent=2)
    samples_path = os.path.join(args.output_dir, "samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump([{"ref": r, "pred": p} for r, p in zip(all_refs[:20], all_preds[:20])],
                  f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
