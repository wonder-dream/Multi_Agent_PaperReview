"""
NER正式训练脚本

训练SciBERT + BiLSTM + CRF命名实体识别模型，支持:
    - SciERC数据集BIO序列标注
    - 早停机制、学习率调度
    - 基于seqeval的span级F1评估
    - 最佳模型保存

Usage:
    python -m train.extraction.train_ner \
        --train_data processed_data/scierc_ner_re/ner_train.jsonl \
        --dev_data processed_data/scierc_ner_re/ner_dev.jsonl \
        --test_data processed_data/scierc_ner_re/ner_test.jsonl \
        --output_dir checkpoints/ner

    python -m train.extraction.train_ner \
        --train_data ... --dev_data ... --output_dir ... \
        --batch_size 16 --lr 5e-5 --epochs 10 --max_length 256
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.extraction.ner_model import SciBERTNERModel
from models.extraction.dataset import NERDataset
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time
from seqeval.metrics import f1_score as seq_f1_score, classification_report as seq_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="训练SciBERT NER模型")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lstm_hidden", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_layers", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm):
    model.train()
    total_loss = 0.0
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
        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return {"loss": total_loss / len(dataloader)}


@torch.no_grad()
def evaluate_ner(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()
        predictions = model.decode(input_ids, attention_mask)
        mask = attention_mask.bool().cpu()
        all_preds.extend([[p for p, m in zip(predictions[i], mask[i]) if m]
                          for i in range(len(predictions))])
        all_labels.extend([[labels[i][j].item() for j, m in enumerate(mask[i]) if m]
                           for i in range(len(labels))])

    id2label = NERDataset.ID2LABEL
    pred_tags = [[id2label.get(t, "O") for t in seq] for seq in all_preds]
    true_tags = [[id2label.get(t, "O") for t in seq] for seq in all_labels]
    return {
        "loss": total_loss / len(dataloader),
        "f1": seq_f1_score(true_tags, pred_tags),
        "report": seq_report(true_tags, pred_tags, output_dict=True)
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
    train_dataset = NERDataset(args.train_data, max_length=args.max_length)
    dev_dataset = NERDataset(args.dev_data, max_length=args.max_length)
    logger.info(f"训练集: {len(train_dataset)} | 验证集: {len(dev_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda x: {
                                  k: torch.stack([d[k] for d in x]) for k in x[0]})
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda x: {
                                k: torch.stack([d[k] for d in x]) for k in x[0]})

    logger.info("正在初始化NER模型...")
    model = SciBERTNERModel(
        model_name=args.model_name, num_labels=len(NERDataset.LABELS),
        lstm_hidden=args.lstm_hidden, dropout_rate=args.dropout,
        freeze_bert_layers=args.freeze_layers
    )
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
    logger.info("开始训练 NER 模型")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args.max_grad_norm)
        dev_metrics = evaluate_ner(model, dev_loader, device)

        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}")
        logger.info(f"[Dev]   Loss: {dev_metrics['loss']:.4f}, Span-F1: {dev_metrics['f1']:.4f}")

        for ent_type, scores in dev_metrics.get("report", {}).items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {ent_type}: F1={scores['f1-score']:.4f}")

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "dev_loss": dev_metrics["loss"],
            "dev_f1": dev_metrics["f1"]
        })

        if args.save_every_epoch:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "f1": dev_metrics["f1"]},
                       os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))

        current_f1 = dev_metrics["f1"]
        if current_f1 > best_f1 + args.min_delta:
            best_f1 = current_f1
            patience_counter = 0
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "f1": best_f1, "config": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"*** 最佳模型已保存 (Span-F1: {best_f1:.4f}) ***")
        else:
            patience_counter += 1
            logger.info(f"早停计数: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"早停触发! 最佳Span-F1: {best_f1:.4f}")
                break

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("在测试集上评估最佳模型")
        logger.info("=" * 60)
        test_dataset = NERDataset(args.test_data, max_length=args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=lambda x: {
                                     k: torch.stack([d[k] for d in x]) for k in x[0]})
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_ner(model, test_loader, device)
        logger.info(f"\n测试集 Span-F1: {test_metrics['f1']:.4f}")
        for ent_type, scores in test_metrics.get("report", {}).items():
            if isinstance(scores, dict) and "f1-score" in scores:
                logger.info(f"  {ent_type}: P={scores.get('precision', 0):.4f}, "
                          f"R={scores.get('recall', 0):.4f}, F1={scores['f1-score']:.4f}")
        result_path = os.path.join(args.output_dir, "test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"f1": test_metrics["f1"], "report": test_metrics["report"]},
                      f, indent=2, ensure_ascii=False)

    logger.info(f"\n训练完成! 最佳验证Span-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
