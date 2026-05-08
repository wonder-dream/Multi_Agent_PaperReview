"""
RE正式训练脚本

训练SciBERT关系分类模型，支持:
    - 实体标记输入 (E1/E2 markers)
    - 类别不平衡处理 (class_weight, NO-RELATION占多数)
    - 早停机制、学习率调度
    - 最佳模型保存

Usage:
    python -m train.extraction.train_re \
        --train_data processed_data/scierc_ner_re/re_train.jsonl \
        --dev_data processed_data/scierc_ner_re/re_dev.jsonl \
        --test_data processed_data/scierc_ner_re/re_test.jsonl \
        --output_dir checkpoints/re

    python -m train.extraction.train_re \
        --train_data ... --dev_data ... --output_dir ... \
        --batch_size 16 --lr 3e-5 --epochs 10 --use_class_weights
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report as sk_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.extraction.re_model import SciBERTRelationClassifier
from models.extraction.dataset import REDataset
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="训练SciBERT关系分类模型")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_layers", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_rel_weight", type=float, default=None)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def compute_class_weights(dataset):
    """计算类别权重"""
    from collections import Counter
    counts = Counter()
    for sample in dataset.samples:
        rel = sample.get("relation", "NO-RELATION")
        counts[rel] += 1
    total = sum(counts.values())
    num_classes = len(REDataset.RELATION_TYPES)
    weights = []
    for rel_name in REDataset.RELATION_TYPES:
        c = counts.get(rel_name, 0)
        w = total / (num_classes * c) if c > 0 else 1.0
        weights.append(w)
    logger.info(f"类别分布: {dict(counts)}")
    logger.info(f"类别权重: {dict(zip(REDataset.RELATION_TYPES, [f'{w:.3f}' for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    }


@torch.no_grad()
def evaluate_re(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    report = sk_report(all_labels, all_preds, target_names=REDataset.RELATION_TYPES,
                       output_dict=True, zero_division=0)
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "report": report
    }


def main():
    args = parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info("正在加载数据集...")
    train_dataset = REDataset(args.train_data, max_length=args.max_length)
    dev_dataset = REDataset(args.dev_data, max_length=args.max_length)
    logger.info(f"训练集: {len(train_dataset)} | 验证集: {len(dev_dataset)}")
    logger.info(f"训练集标签分布: {train_dataset.get_label_distribution()}")

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda x: {
                                  k: torch.stack([d[k] for d in x]) for k in x[0]})
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda x: {
                                k: torch.stack([d[k] for d in x]) for k in x[0]})

    logger.info("正在初始化RE模型...")
    model = SciBERTRelationClassifier(
        model_name=args.model_name, num_relations=len(REDataset.RELATION_TYPES),
        dropout_rate=args.dropout, freeze_bert_layers=args.freeze_layers
    )
    if class_weights is not None:
        ce_weight = class_weights
        original_forward = model.forward

        def weighted_forward(input_ids, attention_mask, labels=None):
            outputs_orig = original_forward(input_ids, attention_mask, labels)
            return outputs_orig
        model._ce_weight = ce_weight
    model.to(device)
    print_model_info(model)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    best_f1 = 0.0
    patience_counter = 0
    training_log = []

    logger.info("=" * 60)
    logger.info("开始训练 RE 模型")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args.max_grad_norm)
        dev_metrics = evaluate_re(model, dev_loader, device)

        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                    f"Macro-F1: {train_metrics['macro_f1']:.4f}")
        logger.info(f"[Dev]   Loss: {dev_metrics['loss']:.4f}, Acc: {dev_metrics['accuracy']:.4f}, "
                    f"Macro-F1: {dev_metrics['macro_f1']:.4f}")

        for rel_name, scores in dev_metrics["report"].items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {rel_name}: F1={scores['f1-score']:.4f}")

        training_log.append({
            "epoch": epoch + 1,
            "train": {k: v for k, v in train_metrics.items()},
            "dev": {k: v for k, v in dev_metrics.items() if k != "report"}
        })

        if args.save_every_epoch:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "macro_f1": dev_metrics["macro_f1"]},
                       os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))

        current_f1 = dev_metrics["macro_f1"]
        if current_f1 > best_f1 + args.min_delta:
            best_f1 = current_f1
            patience_counter = 0
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "macro_f1": best_f1, "config": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"*** 最佳模型已保存 (Macro-F1: {best_f1:.4f}) ***")
        else:
            patience_counter += 1
            logger.info(f"早停计数: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"早停触发! 最佳Macro-F1: {best_f1:.4f}")
                break

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("在测试集上评估最佳模型")
        logger.info("=" * 60)
        test_dataset = REDataset(args.test_data, max_length=args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=lambda x: {
                                     k: torch.stack([d[k] for d in x]) for k in x[0]})
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_re(model, test_loader, device)
        logger.info(f"\n测试集 Accuracy: {test_metrics['accuracy']:.4f}, Macro-F1: {test_metrics['macro_f1']:.4f}")
        for rel_name, scores in test_metrics["report"].items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {rel_name}: P={scores.get('precision', 0):.4f}, "
                          f"R={scores.get('recall', 0):.4f}, F1={scores['f1-score']:.4f}")
        result_path = os.path.join(args.output_dir, "test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": test_metrics["accuracy"], "macro_f1": test_metrics["macro_f1"],
                       "report": test_metrics["report"]}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n训练完成! 最佳验证Macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
