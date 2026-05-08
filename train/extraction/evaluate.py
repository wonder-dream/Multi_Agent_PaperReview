"""
信息抽取评估脚本

加载训练好的NER/RE模型，在测试集上全面评估

Usage:
    python -m train.extraction.evaluate \
        --model_path checkpoints/ner/best_model.pt \
        --model_type ner \
        --test_data processed_data/scierc_ner_re/ner_test.jsonl \
        --output_dir results/ner

    python -m train.extraction.evaluate \
        --model_path checkpoints/re/best_model.pt \
        --model_type re \
        --test_data processed_data/scierc_ner_re/re_test.jsonl \
        --output_dir results/re
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.extraction.ner_model import SciBERTNERModel
from models.extraction.re_model import SciBERTRelationClassifier
from models.extraction.dataset import NERDataset, REDataset
from utils.classifier_utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="评估信息抽取模型")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["ner", "re"])
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_ner(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        predictions = model.decode(input_ids, attention_mask)
        mask = attention_mask.bool().cpu()
        all_preds.extend([[p for p, m in zip(predictions[i], mask[i]) if m]
                          for i in range(len(predictions))])
        all_labels.extend([[labels[i][j].item() for j, m in enumerate(mask[i]) if m]
                           for i in range(len(labels))])

    from seqeval.metrics import f1_score, classification_report
    id2label = NERDataset.ID2LABEL
    pred_tags = [[id2label.get(t, "O") for t in seq] for seq in all_preds]
    true_tags = [[id2label.get(t, "O") for t in seq] for seq in all_labels]
    return {
        "f1": f1_score(true_tags, pred_tags),
        "report": classification_report(true_tags, pred_tags, output_dict=True)
    }


@torch.no_grad()
def evaluate_re(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import f1_score, accuracy_score, classification_report as sk_report
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "report": sk_report(all_labels, all_preds, target_names=REDataset.RELATION_TYPES,
                           output_dict=True, zero_division=0)
    }


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

    if args.model_type == "ner":
        model = SciBERTNERModel(model_name=args.model_name, num_labels=len(NERDataset.LABELS))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        test_dataset = NERDataset(args.test_data, max_length=args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=lambda x: {
                                     k: torch.stack([d[k] for d in x]) for k in x[0]})
        logger.info(f"测试集: {len(test_dataset)} 条")

        metrics = evaluate_ner(model, test_loader, device)

        logger.info("\n" + "=" * 50)
        logger.info("NER 评估结果")
        logger.info("=" * 50)
        logger.info(f"Span-F1: {metrics['f1']:.4f}")
        for ent_type, scores in metrics["report"].items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {ent_type}: P={scores.get('precision', 0):.4f}, "
                          f"R={scores.get('recall', 0):.4f}, F1={scores['f1-score']:.4f}")

    else:
        model = SciBERTRelationClassifier(model_name=args.model_name,
                                           num_relations=len(REDataset.RELATION_TYPES))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        test_dataset = REDataset(args.test_data, max_length=args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=lambda x: {
                                     k: torch.stack([d[k] for d in x]) for k in x[0]})
        logger.info(f"测试集: {len(test_dataset)} 条")
        logger.info(f"标签分布: {test_dataset.get_label_distribution()}")

        metrics = evaluate_re(model, test_loader, device)

        logger.info("\n" + "=" * 50)
        logger.info("RE 评估结果")
        logger.info("=" * 50)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro-F1: {metrics['macro_f1']:.4f}")
        for rel_name, scores in metrics["report"].items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {rel_name}: P={scores.get('precision', 0):.4f}, "
                          f"R={scores.get('recall', 0):.4f}, F1={scores['f1-score']:.4f}")

    result_path = os.path.join(args.output_dir, f"{args.model_type}_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"},
                  f, indent=2, ensure_ascii=False)
    report_path = os.path.join(args.output_dir, f"{args.model_type}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics.get("report", {}), f, indent=2, ensure_ascii=False)

    logger.info(f"\n结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
