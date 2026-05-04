"""
质量分类训练脚本

训练SciBERT质量分类器 (accept vs reject)，支持:
    - 类别不平衡处理 (class_weight)
    - 从PeerRead数据训练
    - 与Domain分类器相同的基础架构

Usage:
    # 使用PeerRead单独数据
    python -m train.classifier.train_quality \
        --train_data processed_data/PeerRead_processed_data/classification/train.jsonl \
        --dev_data processed_data/PeerRead_processed_data/classification/dev.jsonl \
        --output_dir checkpoints/quality_peerread

    # 使用合并数据
    python -m train.classifier.train_quality \
        --train_data processed_data/arxiv_PeerRead_merge_data/classification/quality_train.jsonl \
        --dev_data processed_data/arxiv_PeerRead_merge_data/classification/quality_dev.jsonl \
        --output_dir checkpoints/quality_merged

    # 指定超参数
    python -m train.classifier.train_quality \
        --train_data ... --dev_data ... --output_dir ... \
        --batch_size 32 --lr 2e-5 --epochs 5
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

from models.classifier.scibert_classifier import SciBERTQualityClassifier
from models.classifier.dataset import QualityDataset, create_dataloaders
from utils.metrics import compute_classification_metrics, format_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练SciBERT质量分类器")
    
    # 数据参数
    parser.add_argument("--train_data", type=str, required=True,
                        help="训练集JSONL路径")
    parser.add_argument("--dev_data", type=str, required=True,
                        help="验证集JSONL路径")
    parser.add_argument("--test_data", type=str, default=None,
                        help="测试集JSONL路径 (可选)")
    
    # 模型参数
    parser.add_argument("--model_name", type=str,
                        default="allenai/scibert_scivocab_uncased",
                        help="SciBERT预训练模型名称")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout比率")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="冻结SciBERT前N层")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 类别不平衡参数
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True,
                        help="是否使用类别权重处理不平衡")
    parser.add_argument("--reject_weight", type=float, default=None,
                        help="手动指定reject类权重 (默认自动计算)")
    
    # 早停参数
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_every_epoch", action="store_true")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float
) -> Dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_classification_metrics(
        all_preds, all_labels, 2, ["accept", "reject"]
    )
    metrics["loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_classification_metrics(
        all_preds, all_labels, 2, ["accept", "reject"]
    )
    metrics["loss"] = avg_loss
    
    return metrics


def main():
    args = parse_args()
    
    # 设备设置
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 加载数据集
    logger.info("正在加载数据集...")
    train_dataset = QualityDataset(
        data_path=args.train_data,
        max_length=args.max_length
    )
    dev_dataset = QualityDataset(
        data_path=args.dev_data,
        max_length=args.max_length
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(dev_dataset)}")
    logger.info(f"训练集标签分布: {train_dataset.get_label_distribution()}")
    
    # 计算类别权重 (处理不平衡)
    class_weights = None
    if args.use_class_weights:
        if args.reject_weight is not None:
            class_weights = torch.tensor([1.0, args.reject_weight], dtype=torch.float)
            logger.info(f"使用手动类别权重: accept=1.0, reject={args.reject_weight}")
        else:
            class_weights = train_dataset.get_class_weights()
            logger.info(f"自动计算的类别权重: {class_weights.tolist()}")
        class_weights = class_weights.to(device)
    
    # DataLoader
    train_loader = create_dataloaders(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = create_dataloaders(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # 初始化模型
    logger.info("正在初始化模型...")
    model = SciBERTQualityClassifier(
        model_name=args.model_name,
        num_labels=2,
        dropout_rate=args.dropout,
        freeze_bert_layers=args.freeze_layers,
        class_weights=class_weights
    )
    model.to(device)
    
    # 优化器
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    logger.info(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    
    # 训练
    best_macro_f1 = 0.0
    patience_counter = 0
    training_log = []
    
    logger.info("=" * 60)
    logger.info("开始训练 Quality 分类器")
    logger.info("=" * 60)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.max_grad_norm
        )
        logger.info(f"[Train] Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}, "
                    f"Macro-F1: {train_metrics['macro_f1']:.4f}")
        
        dev_metrics = evaluate(model, dev_loader, device)
        logger.info(f"[Dev]   Loss: {dev_metrics['loss']:.4f}, "
                    f"Acc: {dev_metrics['accuracy']:.4f}, "
                    f"Macro-F1: {dev_metrics['macro_f1']:.4f}")
        
        # 详细指标
        logger.info("\n验证集详细指标:")
        for name, score in dev_metrics["per_class_f1"].items():
            logger.info(f"  {name}: F1={score:.4f}")
        
        training_log.append({
            "epoch": epoch + 1,
            "train": {k: v for k, v in train_metrics.items() if k != "confusion_matrix"},
            "dev": {k: v for k, v in dev_metrics.items() if k != "confusion_matrix"}
        })
        
        if args.save_every_epoch:
            epoch_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "macro_f1": dev_metrics["macro_f1"]
            }, epoch_path)
        
        # 早停
        current_macro_f1 = dev_metrics["macro_f1"]
        if current_macro_f1 > best_macro_f1 + args.min_delta:
            best_macro_f1 = current_macro_f1
            patience_counter = 0
            
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "macro_f1": best_macro_f1,
                "config": vars(args)
            }, best_path)
            logger.info(f"*** 最佳模型已保存 (Macro-F1: {best_macro_f1:.4f}) ***")
        else:
            patience_counter += 1
            logger.info(f"早停计数: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"早停触发! 最佳Macro-F1: {best_macro_f1:.4f}")
                break
    
    # 保存日志
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    # 测试集评估
    if args.test_data and os.path.exists(args.test_data):
        logger.info("\n" + "=" * 60)
        logger.info("在测试集上评估最佳模型")
        logger.info("=" * 60)
        
        test_dataset = QualityDataset(
            data_path=args.test_data,
            max_length=args.max_length
        )
        test_loader = create_dataloaders(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        checkpoint = torch.load(
            os.path.join(args.output_dir, "best_model.pt"),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        test_metrics = evaluate(model, test_loader, device)
        logger.info("\n测试集结果:")
        logger.info(format_metrics(test_metrics, ["accept", "reject"]))
        
        result_path = os.path.join(args.output_dir, "test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
                     f, indent=2, ensure_ascii=False)
    
    logger.info("\n训练完成!")
    logger.info(f"最佳验证Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"模型保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
