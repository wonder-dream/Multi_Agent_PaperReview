"""
分类器 Debug 训练脚本 (适用于本地 4060 快速验证)

功能:
    - 从训练集采样少量数据 (默认100条) 快速验证代码流程
    - 小 batch_size (默认4) 适配 8GB 显存
    - 冻结 SciBERT 层 (只训练分类头) 大幅加速
    - 仅训练 2-3 个 epoch
    - 自动检测 CUDA/CPU 并打印显存信息
    - 支持 domain/quality/multitask 三种模式

Usage:
    # Domain 分类 debug (最轻量)
    python -m train.classifier.debug_train \
        --processed_data_dir processed_data \
        --model_type domain \
        --sample_size 100 \
        --batch_size 4 \
        --epochs 2

    # Quality 分类 debug
    python -m train.classifier.debug_train \
        --processed_data_dir processed_data \
        --model_type quality \
        --sample_size 100 \
        --batch_size 4

    # MultiTask 联合训练 debug
    python -m train.classifier.debug_train \
        --processed_data_dir processed_data \
        --model_type multitask \
        --sample_size 100 \
        --batch_size 4

    # 不解冻BERT层 (更快但精度更低，仅验证代码)
    python -m train.classifier.debug_train \
        --processed_data_dir processed_data \
        --model_type domain \
        --freeze_layers 12 \
        --sample_size 50 \
        --epochs 1

    # 指定设备 (强制CPU测试)
    python -m train.classifier.debug_train ... --device cpu
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.classifier.scibert_classifier import (
    SciBERTDomainClassifier, SciBERTQualityClassifier,
    SciBERTMethodTypeClassifier, SciBERTMultiTaskClassifier
)
from models.classifier.dataset import (
    DomainDataset, QualityDataset, MethodTypeDataset, MultiTaskDataset, create_dataloaders
)
from utils.metrics import compute_classification_metrics, compute_multilabel_metrics, format_metrics
from utils.classifier_utils import set_seed, get_device, print_model_info, format_time


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug训练 (适配4060)")
    
    # 模式选择
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["domain", "quality", "method", "multitask"],
                        help="调试的模型类型")
    
    # 数据目录
    parser.add_argument("--processed_data_dir", type=str, default="processed_data",
                        help="处理后数据的根目录")
    
    # 采样参数 (debug关键)
    parser.add_argument("--sample_size", type=int, default=100,
                        help="从训练集采样的样本数 (默认100)")
    parser.add_argument("--dev_sample_size", type=int, default=30,
                        help="从验证集采样的样本数 (默认30)")
    parser.add_argument("--random_sample", action=argparse.BooleanOptionalAction, default=True,
                        help="随机采样 (True) 或取前N条 (False)")
    
    # 模型参数 (debug模式小配置)
    parser.add_argument("--model_name", type=str,
                        default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=256,
                        help="最大序列长度 (debug模式用256节省显存)")
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="冻结SciBERT前N层 (12=全冻结，只训分类头)")
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # 训练参数 (适配4060)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="batch_size (4060建议4)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率 (冻结层时可适当增大)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="训练轮数 (debug模式2轮即可)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # 多任务参数
    parser.add_argument("--domain_weight", type=float, default=1.0)
    parser.add_argument("--quality_weight", type=float, default=1.0)
    
    # 输出
    parser.add_argument("--output_dir", type=str, default="checkpoints/debug",
                        help="debug输出目录")
    parser.add_argument("--run_name", type=str, default=None,
                        help="本次运行的名称 (默认自动命名)")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="数据加载线程数 (debug建议0，避免多进程问题)")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (auto/cuda/cpu)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="打印详细信息")
    
    return parser.parse_args()


def print_system_info(device: torch.device):
    """打印系统和设备信息"""
    logger.info("=" * 60)
    logger.info("系统信息")
    logger.info("=" * 60)
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA版本: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info(f"总显存: {total_mem:.2f} GB")
        
        # 打印当前显存使用
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        logger.info(f"已分配显存: {allocated:.2f} MB")
        logger.info(f"保留显存: {reserved:.2f} MB")
    else:
        logger.info("使用CPU运行")
    
    logger.info("=" * 60)


def print_gpu_memory(device: torch.device, prefix: str = ""):
    """打印当前GPU显存使用情况"""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        logger.info(f"[显存] {prefix} 已用: {allocated:.1f}MB | 峰值: {peak:.1f}MB")


def sample_dataset(dataset, sample_size: int, random: bool = True, seed: int = 42):
    """
    从数据集中采样少量样本用于debug
    
    Args:
        dataset: 原始数据集
        sample_size: 采样数量
        random: 是否随机采样
        seed: 随机种子
    
    Returns:
        Subset数据集
    """
    total = len(dataset)
    if sample_size >= total:
        logger.info(f"[Debug] 数据集共{total}条，小于采样数{sample_size}，使用全部数据")
        return dataset
    
    if random:
        np.random.seed(seed)
        indices = np.random.choice(total, sample_size, replace=False).tolist()
    else:
        indices = list(range(sample_size))
    
    logger.info(f"[Debug] 从{total}条中采样{sample_size}条")
    return Subset(dataset, indices)


def create_debug_dataloaders(args, device):
    """
    创建debug用的DataLoader
    
    Returns:
        (train_loader, dev_loader, 额外信息dict)
    """
    merge_dir = os.path.join(args.processed_data_dir, "arxiv_PeerRead_merge_data", "classification")
    
    if args.model_type == "domain":
        # Domain数据路径
        train_path = os.path.join(merge_dir, "merged_train.jsonl")
        dev_path = os.path.join(merge_dir, "merged_dev.jsonl")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练数据: {train_path}")
        
        # 加载完整数据集
        train_dataset = DomainDataset(train_path, max_length=args.max_length)
        dev_dataset = DomainDataset(dev_path, max_length=args.max_length)
        
        # 采样
        train_dataset = sample_dataset(train_dataset, args.sample_size, args.random_sample, args.seed)
        dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size, args.random_sample, args.seed)
        
        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                   shuffle=True, num_workers=args.num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        logger.info(f"[Debug Domain] 训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")
        
        return train_loader, dev_loader, {}
    
    elif args.model_type == "quality":
        train_path = os.path.join(merge_dir, "quality_train.jsonl")
        dev_path = os.path.join(merge_dir, "quality_dev.jsonl")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练数据: {train_path}")
        
        train_dataset = QualityDataset(train_path, max_length=args.max_length)
        dev_dataset = QualityDataset(dev_path, max_length=args.max_length)
        
        train_dataset = sample_dataset(train_dataset, args.sample_size, args.random_sample, args.seed)
        dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size, args.random_sample, args.seed)
        
        # 计算class_weights (从原始数据计算，保证准确性)
        full_dataset = QualityDataset(train_path, max_length=args.max_length)
        class_weights = full_dataset.get_class_weights().to(device)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        logger.info(f"[Debug Quality] 训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")
        logger.info(f"[Debug Quality] 类别权重: {class_weights.tolist()}")
        
        return train_loader, dev_loader, {"class_weights": class_weights}
    
    elif args.model_type == "multitask":
        domain_train = os.path.join(merge_dir, "merged_train.jsonl")
        domain_dev = os.path.join(merge_dir, "merged_dev.jsonl")
        quality_train = os.path.join(merge_dir, "quality_train.jsonl")
        quality_dev = os.path.join(merge_dir, "quality_dev.jsonl")
        
        # 先加载完整数据集计算权重
        full_train = MultiTaskDataset(domain_train, quality_train, 
                                       max_length=args.max_length, 
                                       oversample_quality=True)
        domain_weights = full_train.get_domain_class_weights().to(device)
        quality_weights = full_train.get_quality_class_weights().to(device)
        
        # 然后采样
        train_dataset = sample_dataset(full_train, args.sample_size * 2, args.random_sample, args.seed)
        
        dev_dataset = MultiTaskDataset(domain_dev, quality_dev,
                                        max_length=args.max_length,
                                        oversample_quality=False)
        dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size * 2, args.random_sample, args.seed)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        
        logger.info(f"[Debug MultiTask] 训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")

        return train_loader, dev_loader, {
            "domain_class_weights": domain_weights,
            "quality_class_weights": quality_weights
        }

    elif args.model_type == "method":
        train_path = os.path.join(merge_dir, "merged_train.jsonl")
        dev_path = os.path.join(merge_dir, "merged_dev.jsonl")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"找不到训练数据: {train_path}")
        train_dataset = MethodTypeDataset(train_path, max_length=args.max_length)
        dev_dataset = MethodTypeDataset(dev_path, max_length=args.max_length)
        train_dataset = sample_dataset(train_dataset, args.sample_size, args.random_sample, args.seed)
        dev_dataset = sample_dataset(dev_dataset, args.dev_sample_size, args.random_sample, args.seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
        logger.info(f"[Debug Method] 训练: {len(train_dataset)} | 验证: {len(dev_dataset)}")
        return train_loader, dev_loader, {}


def train_epoch_debug(model, dataloader, optimizer, scheduler, device, max_grad_norm, model_type):
    """Debug模式训练一个epoch，带详细显存监控"""
    model.train()
    total_loss = 0.0
    step_times = []
    
    # 按任务收集预测
    all_preds = {"domain": [], "quality": [], "method": []}
    all_labels = {"domain": [], "quality": [], "method": []}

    for batch_idx, batch in enumerate(dataloader):
        step_start = time.time()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()
        
        if model_type == "domain":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            probs = outputs["probs"].detach().cpu().numpy()
            all_preds["domain"].extend(probs.tolist())
            all_labels["domain"].extend(labels.cpu().numpy().tolist())
        
        elif model_type == "quality":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds["quality"].extend(preds)
            all_labels["quality"].extend(labels.cpu().numpy())

        elif model_type == "method":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds["method"].extend(preds)
            all_labels["method"].extend(labels.cpu().numpy())

        elif model_type == "multitask":
            domain_labels = batch["domain_labels"].to(device)
            quality_labels = batch["quality_labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           domain_labels=domain_labels, quality_labels=quality_labels)
            loss = outputs["loss"]

            valid_d = domain_labels.sum(dim=-1) >= 0
            if valid_d.any():
                dprobs = outputs["domain_probs"][valid_d].detach().cpu().numpy()
                all_preds["domain"].extend(dprobs.tolist())
                all_labels["domain"].extend(domain_labels[valid_d].cpu().numpy().tolist())

            valid_q = quality_labels >= 0
            if valid_q.any():
                qp = torch.argmax(outputs["quality_logits"], dim=-1)
                all_preds["quality"].extend(qp[valid_q].cpu().numpy())
                all_labels["quality"].extend(quality_labels[valid_q].cpu().numpy())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        step_times.append(time.time() - step_start)
        
        # 每10步打印显存和速度
        if (batch_idx + 1) % 10 == 0:
            avg_step_time = np.mean(step_times[-10:])
            logger.info(f"  Step {batch_idx+1}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Speed: {avg_step_time:.2f}s/batch")
            print_gpu_memory(device, f"Step {batch_idx+1}")
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算指标
    result = {"loss": avg_loss}
    
    if all_labels["domain"]:
        result["domain"] = compute_multilabel_metrics(
            np.array(all_preds["domain"]), np.array(all_labels["domain"]),
            ["NLP", "CV", "ML", "AI"]
        )
    
    if all_labels["quality"]:
        result["quality"] = compute_classification_metrics(
            np.array(all_preds["quality"]), np.array(all_labels["quality"]),
            3, ["Acceptable", "Borderline", "Weak Reject"]
        )

    if all_labels["method"]:
        result["method"] = compute_classification_metrics(
            np.array(all_preds["method"]), np.array(all_labels["method"]),
            4, ["Empirical", "Theoretical", "Survey", "Benchmark"]
        )

    return result


@torch.no_grad()
def evaluate_debug(model, dataloader, device, model_type):
    """Debug模式评估"""
    model.eval()
    total_loss = 0.0
    
    all_preds = {"domain": [], "quality": [], "method": []}
    all_labels = {"domain": [], "quality": [], "method": []}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        if model_type == "domain":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            probs = outputs["probs"].detach().cpu().numpy()
            all_preds["domain"].extend(probs.tolist())
            all_labels["domain"].extend(labels.cpu().numpy().tolist())
        
        elif model_type == "quality":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds["quality"].extend(preds)
            all_labels["quality"].extend(labels.cpu().numpy())

        elif model_type == "method":
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            all_preds["method"].extend(preds)
            all_labels["method"].extend(labels.cpu().numpy())

        elif model_type == "multitask":
            domain_labels = batch["domain_labels"].to(device)
            quality_labels = batch["quality_labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           domain_labels=domain_labels, quality_labels=quality_labels)
            loss = outputs["loss"]
            
            valid_d = domain_labels.sum(dim=-1) >= 0
            if valid_d.any():
                dprobs = outputs["domain_probs"][valid_d].detach().cpu().numpy()
                all_preds["domain"].extend(dprobs.tolist())
                all_labels["domain"].extend(domain_labels[valid_d].cpu().numpy().tolist())
            
            valid_q = quality_labels >= 0
            if valid_q.any():
                qp = torch.argmax(outputs["quality_logits"], dim=-1)
                all_preds["quality"].extend(qp[valid_q].cpu().numpy())
                all_labels["quality"].extend(quality_labels[valid_q].cpu().numpy())
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    result = {"loss": avg_loss}
    
    if all_labels["domain"]:
        result["domain"] = compute_multilabel_metrics(
            np.array(all_preds["domain"]), np.array(all_labels["domain"]),
            ["NLP", "CV", "ML", "AI"]
        )
    if all_labels["quality"]:
        result["quality"] = compute_classification_metrics(
            np.array(all_preds["quality"]), np.array(all_labels["quality"]),
            3, ["Acceptable", "Borderline", "Weak Reject"]
        )

    if all_labels["method"]:
        result["method"] = compute_classification_metrics(
            np.array(all_preds["method"]), np.array(all_labels["method"]),
            4, ["Empirical", "Theoretical", "Survey", "Benchmark"]
        )

    return result


def main():
    args = parse_args()
    
    # 自动命名
    if args.run_name is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        freeze_tag = f"_freeze{args.freeze_layers}" if args.freeze_layers > 0 else ""
        args.run_name = f"debug_{args.model_type}_{timestamp}{freeze_tag}"
    
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, "debug_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    # 设置种子和设备
    set_seed(args.seed)
    device = get_device(args.device)
    
    # 打印系统信息
    print_system_info(device)
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Debug 训练 - {args.model_type.upper()}")
    logger.info(f"运行名称: {args.run_name}")
    logger.info(f"采样: {args.sample_size}条训练 / {args.dev_sample_size}条验证")
    logger.info(f"序列长度: {args.max_length} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    logger.info(f"冻结层数: {args.freeze_layers} (0=全部可训练, 12=全冻结)")
    logger.info("=" * 60)
    
    # 创建数据加载器
    logger.info("\n[1/5] 加载数据...")
    try:
        train_loader, dev_loader, extra_info = create_debug_dataloaders(args, device)
        logger.info("数据加载成功!")
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        logger.info("请确认数据路径正确，或运行数据预处理脚本:")
        logger.info("  processed_data/arxiv_PeerRead_merge_data/classification/")
        return
    
    # 创建模型
    logger.info(f"\n[2/5] 创建 {args.model_type} 模型...")
    if args.model_type == "domain":
        model = SciBERTDomainClassifier(
            model_name=args.model_name,
            freeze_bert_layers=args.freeze_layers,
            dropout_rate=args.dropout
        )
    elif args.model_type == "quality":
        class_weights = extra_info.get("class_weights")
        model = SciBERTQualityClassifier(
            model_name=args.model_name,
            freeze_bert_layers=args.freeze_layers,
            dropout_rate=args.dropout,
            class_weights=class_weights
        )
    elif args.model_type == "method":
        model = SciBERTMethodTypeClassifier(
            model_name=args.model_name,
            freeze_bert_layers=args.freeze_layers,
            dropout_rate=args.dropout
        )
    elif args.model_type == "multitask":
        model = SciBERTMultiTaskClassifier(
            model_name=args.model_name,
            freeze_bert_layers=args.freeze_layers,
            dropout_rate=args.dropout,
            domain_class_weights=extra_info.get("domain_class_weights"),
            quality_class_weights=extra_info.get("quality_class_weights"),
            task_weights=[args.domain_weight, args.quality_weight]
        )
    
    model.to(device)
    print_model_info(model)
    print_gpu_memory(device, "模型加载后")
    
    # 优化器
    logger.info(f"\n[3/5] 准备训练...")
    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    logger.info(f"总步数: {total_steps} ({len(train_loader)}步/epoch x {args.epochs}epochs)")
    
    # 训练循环
    logger.info(f"\n[4/5] 开始训练...")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 训练
        train_metrics = train_epoch_debug(
            model, train_loader, optimizer, scheduler, 
            device, args.max_grad_norm, args.model_type
        )
        
        # 评估
        dev_metrics = evaluate_debug(model, dev_loader, device, args.model_type)
        
        epoch_time = time.time() - epoch_start
        
        # 打印结果
        logger.info(f"\n  Epoch {epoch+1} 结果 (用时: {format_time(epoch_time)}):")
        logger.info(f"  [Train] Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  [Dev]   Loss: {dev_metrics['loss']:.4f}")
        
        if "domain" in dev_metrics:
            logger.info(f"  [Dev]   Domain Micro-F1: {dev_metrics['domain']['micro_f1']:.4f} | "
                        f"Macro-F1: {dev_metrics['domain']['macro_f1']:.4f}")
            for name, score in dev_metrics["domain"]["per_class_f1"].items():
                logger.info(f"          {name}: F1={score:.4f}")
        
        if "quality" in dev_metrics:
            logger.info(f"  [Dev]   Quality Acc: {dev_metrics['quality']['accuracy']:.4f} | "
                        f"Macro-F1: {dev_metrics['quality']['macro_f1']:.4f}")
            for name, score in dev_metrics["quality"]["per_class_f1"].items():
                logger.info(f"          {name}: F1={score:.4f}")

        if "method" in dev_metrics:
            logger.info(f"  [Dev]   Method Acc: {dev_metrics['method']['accuracy']:.4f} | "
                        f"Macro-F1: {dev_metrics['method']['macro_f1']:.4f}")
            for name, score in dev_metrics["method"]["per_class_f1"].items():
                logger.info(f"          {name}: F1={score:.4f}")

        print_gpu_memory(device, f"Epoch {epoch+1} 结束")
    
    total_time = time.time() - start_time
    
    # 保存最终模型
    logger.info(f"\n[5/5] 保存模型...")
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "training_time": total_time
    }, final_path)
    logger.info(f"模型已保存: {final_path}")
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("Debug 训练完成!")
    logger.info("=" * 60)
    logger.info(f"总用时: {format_time(total_time)}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型文件: {final_path}")
    
    # 验证推理
    logger.info("\n--- 推理验证 ---")
    model.eval()
    test_text = "Title: BERT: Pre-training of Deep Bidirectional Transformers. " \
                "Abstract: We introduce a new language representation model called BERT."
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoding = tokenizer(test_text, max_length=args.max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"].to(device),
                       attention_mask=encoding["attention_mask"].to(device))
    
    if args.model_type == "domain":
        probs = torch.sigmoid(outputs["logits"])[0].cpu().numpy()
        labels = ["NLP", "CV", "ML", "AI"]
        active = [l for l, p in zip(labels, probs) if p >= 0.5] or [labels[probs.argmax()]]
        logger.info(f"测试文本: '{test_text[:80]}...'")
        logger.info(f"预测领域: {active} (阈值=0.5)")
        for l, p in zip(labels, probs):
            logger.info(f"  {l}: {p:.4f}")
    
    elif args.model_type == "quality":
        probs = torch.softmax(outputs["logits"], dim=-1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        labels = ["Acceptable", "Borderline", "Weak Reject"]
        logger.info(f"测试文本: '{test_text[:80]}...'")
        logger.info(f"预测质量: {labels[pred_idx]} (置信度: {probs[pred_idx]:.4f})")

    elif args.model_type == "method":
        probs = torch.softmax(outputs["logits"], dim=-1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        labels = ["Empirical", "Theoretical", "Survey", "Benchmark"]
        logger.info(f"测试文本: '{test_text[:80]}...'")
        logger.info(f"预测方法类型: {labels[pred_idx]} (置信度: {probs[pred_idx]:.4f})")
        for l, p in zip(labels, probs):
            logger.info(f"  {l}: {p:.4f}")

    elif args.model_type == "multitask":
        d_probs = torch.sigmoid(outputs["domain_logits"])[0].cpu().numpy()
        q_probs = torch.softmax(outputs["quality_logits"], dim=-1)[0].cpu().numpy()
        d_labels = ["NLP", "CV", "ML", "AI"]
        q_labels = ["Acceptable", "Borderline", "Weak Reject"]

        d_active = [l for l, p in zip(d_labels, d_probs) if p >= 0.5] or [d_labels[d_probs.argmax()]]
        logger.info(f"测试文本: '{test_text[:80]}...'")
        logger.info(f"预测领域: {d_active} (阈值=0.5)")
        logger.info(f"预测质量: {q_labels[q_probs.argmax()]} (置信度: {q_probs.max():.4f})")
    
    logger.info("\n一切正常! 可以部署到 5090 进行正式训练。")
    print_gpu_memory(device, "最终")


if __name__ == "__main__":
    main()
