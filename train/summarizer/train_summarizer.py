"""
摘要生成正式训练脚本

训练 BART 抽象式摘要模型:
    - SciTLDR 数据集
    - ROUGE/BERTScore 评估
    - 早停机制、学习率调度
    - 最佳模型保存

Usage:
    python -m train.summarizer.train_summarizer \
        --train_data processed_data/sciTLDR_data/train.jsonl \
        --dev_data processed_data/sciTLDR_data/dev.jsonl \
        --test_data processed_data/sciTLDR_data/test.jsonl \
        --output_dir checkpoints/summarizer

    python -m train.summarizer.train_summarizer \
        --model_name facebook/bart-large \
        --batch_size 4 --lr 3e-5 --epochs 5
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from rouge import Rouge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.summarizer.generative import BARTSummarizer
from models.summarizer.dataset import SciTLDRDataset
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

rouge = Rouge()


def parse_args():
    parser = argparse.ArgumentParser(description="训练BART摘要模型")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--freeze_decoder", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=0.01)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every_epoch", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == model.bart.config.pad_token_id] = -100

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return {"loss": total_loss / len(dataloader)}


@torch.no_grad()
def evaluate(model, dataloader, device, tokenizer, max_target_length):
    model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels_clean = labels.clone()
        labels_clean[labels_clean == model.bart.config.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_clean)
        total_loss += outputs["loss"].item()

        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   max_length=max_target_length, num_beams=4)
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
        all_preds.extend(preds)
        all_refs.extend(refs)

    metrics = {"loss": total_loss / len(dataloader)}
    try:
        scores = rouge.get_scores(all_preds, all_refs, avg=True)
        metrics["rouge-1"] = scores["rouge-1"]["f"]
        metrics["rouge-2"] = scores["rouge-2"]["f"]
        metrics["rouge-l"] = scores["rouge-l"]["f"]
    except Exception:
        metrics["rouge-1"] = metrics["rouge-2"] = metrics["rouge-l"] = 0.0
    return metrics


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

    logger.info("正在加载 SciTLDR 数据集...")
    train_dataset = SciTLDRDataset(args.train_data, tokenizer_name=args.model_name,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length)
    dev_dataset = SciTLDRDataset(args.dev_data, tokenizer_name=args.model_name,
                                  max_source_length=args.max_source_length,
                                  max_target_length=args.max_target_length)
    logger.info(f"训练集: {len(train_dataset)} | 验证集: {len(dev_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda x: {
                                  k: torch.stack([d[k] for d in x]) for k in x[0]})
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda x: {
                                k: torch.stack([d[k] for d in x]) for k in x[0]})

    logger.info("正在初始化 BART 模型...")
    model = BARTSummarizer(model_name=args.model_name, max_target_length=args.max_target_length,
                            freeze_encoder_layers=args.freeze_encoder,
                            freeze_decoder_layers=args.freeze_decoder)
    model.to(device)
    print_model_info(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    best_rouge_l = 0.0
    patience_counter = 0
    training_log = []

    logger.info("=" * 60)
    logger.info("开始训练 BART 摘要模型")
    logger.info("=" * 60)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, args.max_grad_norm)
        dev_metrics = evaluate(model, dev_loader, device, tokenizer, args.max_target_length)

        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}")
        logger.info(f"[Dev]   Loss: {dev_metrics['loss']:.4f}, "
                    f"ROUGE-1: {dev_metrics.get('rouge-1', 0):.4f}, "
                    f"ROUGE-2: {dev_metrics.get('rouge-2', 0):.4f}, "
                    f"ROUGE-L: {dev_metrics.get('rouge-l', 0):.4f}")

        training_log.append({"epoch": epoch + 1, "train": train_metrics, "dev": dev_metrics})

        if args.save_every_epoch:
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "rouge_l": dev_metrics.get("rouge-l", 0)},
                       os.path.join(args.output_dir, f"epoch_{epoch+1}.pt"))

        current_rouge = dev_metrics.get("rouge-l", 0)
        if current_rouge > best_rouge_l + args.min_delta:
            best_rouge_l = current_rouge
            patience_counter = 0
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "rouge_l": best_rouge_l, "config": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"*** 最佳模型已保存 (ROUGE-L: {best_rouge_l:.4f}) ***")
        else:
            patience_counter += 1
            logger.info(f"早停计数: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"早停触发! 最佳 ROUGE-L: {best_rouge_l:.4f}")
                break

        logger.info(f"Epoch 用时: {format_time(time.time() - epoch_start)}")

    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)

    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("在测试集上评估最佳模型")
        logger.info("=" * 60)
        test_dataset = SciTLDRDataset(args.test_data, tokenizer_name=args.model_name,
                                       max_source_length=args.max_source_length,
                                       max_target_length=args.max_target_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, collate_fn=lambda x: {
                                      k: torch.stack([d[k] for d in x]) for k in x[0]})
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, tokenizer, args.max_target_length)
        logger.info(f"\n测试集: ROUGE-1={test_metrics.get('rouge-1', 0):.4f}, "
                    f"ROUGE-2={test_metrics.get('rouge-2', 0):.4f}, "
                    f"ROUGE-L={test_metrics.get('rouge-l', 0):.4f}")
        with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"\n训练完成! 最佳 ROUGE-L: {best_rouge_l:.4f}")


if __name__ == "__main__":
    main()
