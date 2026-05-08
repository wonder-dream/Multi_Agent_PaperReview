"""
摘要生成 Debug 训练脚本

快速验证 BART 摘要模型的代码流程:
    - 从 SciTLDR 采样少量数据
    - 小 batch_size 适配本地 GPU
    - 可选冻结 encoder/decoder 层加速
    - 训练 2-3 epoch 验证损失下降

Usage:
    python -m train.summarizer.debug_train --sample_size 100 --batch_size 2 --epochs 2
    python -m train.summarizer.debug_train --sample_size 50 --freeze_encoder 6 --freeze_decoder 6
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.summarizer.generative import BARTSummarizer
from models.summarizer.dataset import SciTLDRDataset
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug训练 (摘要生成)")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--freeze_decoder", type=int, default=0)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--dev_sample_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="checkpoints/debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def sample_dataset(dataset, sample_size, seed=42):
    total = len(dataset)
    if sample_size >= total:
        return dataset
    np.random.seed(seed)
    indices = np.random.choice(total, sample_size, replace=False).tolist()
    return Subset(dataset, indices)


def train_epoch(model, dataloader, optimizer, scheduler, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
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
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Step {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return {"loss": total_loss / len(dataloader)}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        labels[labels == model.bart.config.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs["loss"].item()

    return {"loss": total_loss / len(dataloader)}


def main():
    args = parse_args()

    if args.run_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.run_name = f"debug_summarizer_{timestamp}"

    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info(f"Debug 训练 - 摘要生成 (BART)")
    logger.info(f"采样: {args.sample_size}条训练 / {args.dev_sample_size}条验证")
    logger.info(f"序列长度: {args.max_source_length} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    logger.info("=" * 60)

    data_dir = os.path.join(args.processed_data_dir, "sciTLDR_data")
    logger.info(f"[1/5] 加载数据 from {data_dir}...")

    train_path = os.path.join(data_dir, "train.jsonl")
    dev_path = os.path.join(data_dir, "dev.jsonl")

    if not os.path.exists(train_path):
        logger.error(f"训练数据不存在: {train_path}")
        logger.info("请确保已运行 SciTLDR 数据预处理")
        return

    train_dataset = SciTLDRDataset(train_path, tokenizer_name=args.model_name,
                                    max_source_length=args.max_source_length,
                                    max_target_length=args.max_target_length)
    dev_dataset = SciTLDRDataset(dev_path, tokenizer_name=args.model_name,
                                  max_source_length=args.max_source_length,
                                  max_target_length=args.max_target_length)

    train_dataset = sample_dataset(train_dataset, args.sample_size, args.seed)
    dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size, args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=lambda x: {
                                  k: torch.stack([d[k] for d in x]) for k in x[0]})
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=lambda x: {
                                k: torch.stack([d[k] for d in x]) for k in x[0]})
    logger.info(f"训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")

    logger.info(f"[2/5] 创建 BART 模型 ({args.model_name})...")
    model = BARTSummarizer(
        model_name=args.model_name,
        max_target_length=args.max_target_length,
        freeze_encoder_layers=args.freeze_encoder,
        freeze_decoder_layers=args.freeze_decoder
    )
    model.to(device)
    print_model_info(model)

    logger.info(f"[3/5] 准备训练...")
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    logger.info(f"总步数: {total_steps}")

    logger.info("[4/5] 开始训练...")
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_metrics = train_epoch(model, train_loader, optimizer, scheduler,
                                     device, args.max_grad_norm)
        dev_metrics = evaluate(model, dev_loader, device)

        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Dev Loss:   {dev_metrics['loss']:.4f}")

        if dev_metrics["loss"] < best_loss:
            best_loss = dev_metrics["loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args)},
                       os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"  *** 最佳模型已保存 (Loss: {best_loss:.4f}) ***")

    logger.info(f"总用时: {format_time(time.time() - start_time)}")

    logger.info("[5/5] 推理验证...")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_text = "We introduce a new language representation model called BERT, which stands for " \
                "Bidirectional Encoder Representations from Transformers. Unlike recent language " \
                "representation models, BERT is designed to pre-train deep bidirectional representations " \
                "from unlabeled text by jointly conditioning on both left and right context."

    encoding = tokenizer(test_text, max_length=args.max_source_length, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             max_length=64, num_beams=4)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"源文本: '{test_text[:120]}...'")
    logger.info(f"生成摘要: '{summary}'")

    logger.info(f"\n模型已保存: {args.output_dir}")
    logger.info("一切正常!")


if __name__ == "__main__":
    main()
