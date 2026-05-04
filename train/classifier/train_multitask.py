"""
多任务联合训练脚本

同时训练Domain分类和Quality分类，共享SciBERT编码器:
    - Domain任务: 4类分类 (NLP, CV, ML, AI)
    - Quality任务: 2类分类 (accept, reject)

通过共享编码器，两个任务互相促进:
    - Domain信息帮助判断质量 (不同领域质量标准不同)
    - Quality信号帮助学习更好的领域表示

Usage:
    python -m train.classifier.train_multitask \
        --domain_train processed_data/arxiv_PeerRead_merge_data/classification/merged_train.jsonl \
        --domain_dev processed_data/arxiv_PeerRead_merge_data/classification/merged_dev.jsonl \
        --quality_train processed_data/arxiv_PeerRead_merge_data/classification/quality_train.jsonl \
        --quality_dev processed_data/arxiv_PeerRead_merge_data/classification/quality_dev.jsonl \
        --output_dir checkpoints/multitask

    # 指定任务权重和超参数
    python -m train.classifier.train_multitask \
        --domain_train ... --domain_dev ... \
        --quality_train ... --quality_dev ... \
        --output_dir ... \
        --domain_weight 1.0 --quality_weight 0.8 \
        --batch_size 16 --lr 2e-5 --epochs 5
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

from models.classifier.scibert_classifier import SciBERTMultiTaskClassifier
from models.classifier.dataset import MultiTaskDataset, create_dataloaders
from utils.metrics import compute_classification_metrics, format_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多任务联合训练")
    
    # Domain数据路径
    parser.add_argument("--domain_train", type=str, required=True,
                        help="Domain训练集JSONL路径")
    parser.add_argument("--domain_dev", type=str, required=True,
                        help="Domain验证集JSONL路径")
    parser.add_argument("--domain_test", type=str, default=None)
    
    # Quality数据路径
    parser.add_argument("--quality_train", type=str, required=True,
                        help="Quality训练集JSONL路径")
    parser.add_argument("--quality_dev", type=str, required=True,
                        help="Quality验证集JSONL路径")
    parser.add_argument("--quality_test", type=str, default=None)
    
    # 模型参数
    parser.add_argument("--model_name", type=str,
                        default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_layers", type=int, default=0)
    
    # 多任务参数
    parser.add_argument("--domain_weight", type=float, default=1.0,
                        help="Domain任务损失权重")
    parser.add_argument("--quality_weight", type=float, default=1.0,
                        help="Quality任务损失权重")
    parser.add_argument("--oversample_quality", action=argparse.BooleanOptionalAction, default=True,
                        help="对quality样本进行过采样以平衡任务")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # 早停
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    
    # 输出
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
    """
    训练一个epoch
    
    多任务训练: 每个batch可能来自domain或quality任务
    通过交替采样实现
    """
    model.train()
    total_loss = 0.0
    total_domain_loss = 0.0
    total_quality_loss = 0.0
    
    domain_preds = []
    domain_labels = []
    quality_preds = []
    quality_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels_batch = batch["domain_labels"].to(device)
        quality_labels_batch = batch["quality_labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_labels=domain_labels_batch,
            quality_labels=quality_labels_batch
        )
        
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        
        if "domain_loss" in outputs:
            total_domain_loss += outputs["domain_loss"].item()
            # 只统计有效的domain标签 (非-1)
            valid_domain = domain_labels_batch >= 0
            if valid_domain.any():
                dpreds = torch.argmax(outputs["domain_logits"], dim=-1)
                domain_preds.extend(dpreds[valid_domain].cpu().numpy())
                domain_labels.extend(domain_labels_batch[valid_domain].cpu().numpy())
        
        if "quality_loss" in outputs:
            total_quality_loss += outputs["quality_loss"].item()
            valid_quality = quality_labels_batch >= 0
            if valid_quality.any():
                qpreds = torch.argmax(outputs["quality_logits"], dim=-1)
                quality_preds.extend(qpreds[valid_quality].cpu().numpy())
                quality_labels.extend(quality_labels_batch[valid_quality].cpu().numpy())
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(dataloader)}, "
                        f"Loss: {loss.item():.4f}")
    
    # 计算指标
    result = {"loss": total_loss / len(dataloader)}
    
    if domain_labels:
        result["domain"] = compute_classification_metrics(
            np.array(domain_preds), np.array(domain_labels), 4, ["NLP", "CV", "ML", "AI"]
        )
        result["domain"]["loss"] = total_domain_loss / len(dataloader)
    
    if quality_labels:
        result["quality"] = compute_classification_metrics(
            np.array(quality_preds), np.array(quality_labels), 2, ["accept", "reject"]
        )
        result["quality"]["loss"] = total_quality_loss / len(dataloader)
    
    return result


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict:
    """评估多任务模型"""
    model.eval()
    total_loss = 0.0
    total_domain_loss = 0.0
    total_quality_loss = 0.0
    
    domain_preds = []
    domain_labels_list = []
    quality_preds = []
    quality_labels_list = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels_batch = batch["domain_labels"].to(device)
        quality_labels_batch = batch["quality_labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            domain_labels=domain_labels_batch,
            quality_labels=quality_labels_batch
        )
        
        total_loss += outputs["loss"].item()
        
        if "domain_loss" in outputs:
            total_domain_loss += outputs["domain_loss"].item()
            valid_domain = domain_labels_batch >= 0
            if valid_domain.any():
                dpreds = torch.argmax(outputs["domain_logits"], dim=-1)
                domain_preds.extend(dpreds[valid_domain].cpu().numpy())
                domain_labels_list.extend(domain_labels_batch[valid_domain].cpu().numpy())
        
        if "quality_loss" in outputs:
            total_quality_loss += outputs["quality_loss"].item()
            valid_quality = quality_labels_batch >= 0
            if valid_quality.any():
                qpreds = torch.argmax(outputs["quality_logits"], dim=-1)
                quality_preds.extend(qpreds[valid_quality].cpu().numpy())
                quality_labels_list.extend(quality_labels_batch[valid_quality].cpu().numpy())
    
    result = {"loss": total_loss / len(dataloader)}
    
    if domain_labels_list:
        result["domain"] = compute_classification_metrics(
            np.array(domain_preds), np.array(domain_labels_list), 4, ["NLP", "CV", "ML", "AI"]
        )
        result["domain"]["loss"] = total_domain_loss / len(dataloader)
    
    if quality_labels_list:
        result["quality"] = compute_classification_metrics(
            np.array(quality_preds), np.array(quality_labels_list), 2, ["accept", "reject"]
        )
        result["quality"]["loss"] = total_quality_loss / len(dataloader)
    
    return result


def main():
    args = parse_args()
    
    # 设备
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
    
    # 加载多任务数据集
    logger.info("正在加载多任务数据集...")
    train_dataset = MultiTaskDataset(
        domain_data_path=args.domain_train,
        quality_data_path=args.quality_train,
        max_length=args.max_length,
        oversample_quality=args.oversample_quality
    )
    dev_dataset = MultiTaskDataset(
        domain_data_path=args.domain_dev,
        quality_data_path=args.quality_dev,
        max_length=args.max_length,
        oversample_quality=False  # 验证集不过采样
    )
    
    logger.info(f"训练集有效长度: {len(train_dataset)}")
    logger.info(f"验证集有效长度: {len(dev_dataset)}")
    
    # 计算类别权重
    domain_class_weights = train_dataset.get_domain_class_weights().to(device)
    quality_class_weights = train_dataset.get_quality_class_weights().to(device)
    
    # DataLoader
    train_loader = create_dataloaders(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = create_dataloaders(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    
    # 初始化多任务模型
    logger.info("正在初始化多任务模型...")
    task_weights = [args.domain_weight, args.quality_weight]
    logger.info(f"任务权重: Domain={args.domain_weight}, Quality={args.quality_weight}")
    
    model = SciBERTMultiTaskClassifier(
        model_name=args.model_name,
        num_domain_labels=4,
        num_quality_labels=2,
        dropout_rate=args.dropout,
        freeze_bert_layers=args.freeze_layers,
        domain_class_weights=domain_class_weights,
        quality_class_weights=quality_class_weights,
        task_weights=task_weights
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
    best_combined_f1 = 0.0
    patience_counter = 0
    training_log = []
    
    logger.info("=" * 60)
    logger.info("开始多任务联合训练")
    logger.info("=" * 60)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 40)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.max_grad_norm
        )
        
        logger.info(f"[Train] Total Loss: {train_metrics['loss']:.4f}")
        if "domain" in train_metrics:
            logger.info(f"[Train] Domain - Acc: {train_metrics['domain']['accuracy']:.4f}, "
                        f"Macro-F1: {train_metrics['domain']['macro_f1']:.4f}")
        if "quality" in train_metrics:
            logger.info(f"[Train] Quality - Acc: {train_metrics['quality']['accuracy']:.4f}, "
                        f"Macro-F1: {train_metrics['quality']['macro_f1']:.4f}")
        
        dev_metrics = evaluate(model, dev_loader, device)
        
        logger.info(f"\n[Dev] Total Loss: {dev_metrics['loss']:.4f}")
        if "domain" in dev_metrics:
            logger.info(f"[Dev] Domain - Acc: {dev_metrics['domain']['accuracy']:.4f}, "
                        f"Macro-F1: {dev_metrics['domain']['macro_f1']:.4f}")
            for name, score in dev_metrics["domain"]["per_class_f1"].items():
                logger.info(f"  Domain {name}: F1={score:.4f}")
        
        if "quality" in dev_metrics:
            logger.info(f"[Dev] Quality - Acc: {dev_metrics['quality']['accuracy']:.4f}, "
                        f"Macro-F1: {dev_metrics['quality']['macro_f1']:.4f}")
            for name, score in dev_metrics["quality"]["per_class_f1"].items():
                logger.info(f"  Quality {name}: F1={score:.4f}")
        
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
            }, epoch_path)
        
        # 早停: 使用Domain和Quality的Macro-F1平均值作为指标
        domain_f1 = dev_metrics.get("domain", {}).get("macro_f1", 0.0)
        quality_f1 = dev_metrics.get("quality", {}).get("macro_f1", 0.0)
        combined_f1 = (domain_f1 + quality_f1) / 2
        
        if combined_f1 > best_combined_f1 + args.min_delta:
            best_combined_f1 = combined_f1
            patience_counter = 0
            
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "combined_f1": best_combined_f1,
                "domain_f1": domain_f1,
                "quality_f1": quality_f1,
                "config": vars(args)
            }, best_path)
            logger.info(f"*** 最佳模型已保存 (Combined F1: {best_combined_f1:.4f}) ***")
        else:
            patience_counter += 1
            logger.info(f"早停计数: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logger.info(f"早停触发! 最佳Combined F1: {best_combined_f1:.4f}")
                break
    
    # 保存日志
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    logger.info("\n训练完成!")
    logger.info(f"最佳Combined F1: {best_combined_f1:.4f}")
    logger.info(f"模型保存在: {args.output_dir}")
    
    # 测试集评估
    if (args.domain_test and os.path.exists(args.domain_test)) or \
       (args.quality_test and os.path.exists(args.quality_test)):
        logger.info("\n" + "=" * 60)
        logger.info("在测试集上评估最佳模型")
        logger.info("=" * 60)
        
        test_dataset = MultiTaskDataset(
            domain_data_path=args.domain_test or args.domain_dev,
            quality_data_path=args.quality_test or args.quality_dev,
            max_length=args.max_length,
            oversample_quality=False
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
        
        if "domain" in test_metrics:
            logger.info("\nDomain测试集结果:")
            logger.info(format_metrics(test_metrics["domain"], ["NLP", "CV", "ML", "AI"]))
        
        if "quality" in test_metrics:
            logger.info("\nQuality测试集结果:")
            logger.info(format_metrics(test_metrics["quality"], ["accept", "reject"]))
        
        result_path = os.path.join(args.output_dir, "test_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
                     f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
