"""
分类器评估脚本

加载训练好的模型，在测试集上进行全面评估，输出:
    - 准确率、Macro-F1、Weighted-F1
    - 每个类别的精确率、召回率、F1
    - 混淆矩阵 (可选保存为图片)

Usage:
    # 评估Domain模型
    python -m train.classifier.evaluate \
        --model_path checkpoints/domain_merged/best_model.pt \
        --model_type domain \
        --test_data processed_data/arxiv_PeerRead_merge_data/classification/merged_test.jsonl \
        --output_dir results/domain

    # 评估Quality模型
    python -m train.classifier.evaluate \
        --model_path checkpoints/quality_merged/best_model.pt \
        --model_type quality \
        --test_data processed_data/arxiv_PeerRead_merge_data/classification/quality_test.jsonl \
        --output_dir results/quality

    # 评估MultiTask模型
    python -m train.classifier.evaluate \
        --model_path checkpoints/multitask/best_model.pt \
        --model_type multitask \
        --domain_test processed_data/arxiv_PeerRead_merge_data/classification/merged_test.jsonl \
        --quality_test processed_data/arxiv_PeerRead_merge_data/classification/quality_test.jsonl \
        --output_dir results/multitask
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

from models.classifier.scibert_classifier import (
    SciBERTDomainClassifier, SciBERTQualityClassifier, SciBERTMultiTaskClassifier
)
from models.classifier.dataset import DomainDataset, QualityDataset, MultiTaskDataset, create_dataloaders
from utils.metrics import compute_classification_metrics, compute_multilabel_metrics, format_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="评估分类器")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["domain", "quality", "multitask"],
                        help="模型类型")
    
    # 数据参数 (单任务)
    parser.add_argument("--test_data", type=str, default=None,
                        help="测试集路径 (domain/quality单任务)")
    
    # 数据参数 (多任务)
    parser.add_argument("--domain_test", type=str, default=None)
    parser.add_argument("--quality_test", type=str, default=None)
    
    # 模型配置
    parser.add_argument("--model_name", type=str,
                        default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 输出
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_confusion_matrix", action="store_true",
                        help="保存混淆矩阵图片")
    
    # 设备
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_domain(model, dataloader, device):
    """评估Domain模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs["probs"].detach().cpu().numpy()

        all_preds.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return compute_multilabel_metrics(
        np.array(all_preds), np.array(all_labels), ["NLP", "CV", "ML", "AI"]
    )


@torch.no_grad()
def evaluate_quality(model, dataloader, device):
    """评估Quality模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    return compute_classification_metrics(
        np.array(all_preds), np.array(all_labels), 2, ["Acceptable", "Borderline", "Weak Reject"]
    )


@torch.no_grad()
def evaluate_multitask(model, dataloader, device):
    """评估MultiTask模型"""
    model.eval()
    domain_preds = []
    domain_labels = []
    quality_preds = []
    quality_labels = []
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_labels_batch = batch["domain_labels"].to(device)
        quality_labels_batch = batch["quality_labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Domain
        valid_domain = domain_labels_batch >= 0
        if valid_domain.any():
            dpreds = torch.argmax(outputs["domain_logits"], dim=-1)
            domain_preds.extend(dpreds[valid_domain].cpu().numpy())
            domain_labels.extend(domain_labels_batch[valid_domain].cpu().numpy())
        
        # Quality
        valid_quality = quality_labels_batch >= 0
        if valid_quality.any():
            qpreds = torch.argmax(outputs["quality_logits"], dim=-1)
            quality_preds.extend(qpreds[valid_quality].cpu().numpy())
            quality_labels.extend(quality_labels_batch[valid_quality].cpu().numpy())
    
    result = {}
    if domain_labels:
        result["domain"] = compute_classification_metrics(
            np.array(domain_preds), np.array(domain_labels), 4, ["NLP", "CV", "ML", "AI"]
        )
    if quality_labels:
        result["quality"] = compute_classification_metrics(
            np.array(quality_preds), np.array(quality_labels), 2, ["Acceptable", "Borderline", "Weak Reject"]
        )
    
    return result


def save_confusion_matrix(cm, class_names, save_path):
    """保存混淆矩阵图片"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"混淆矩阵已保存到 {save_path}")
    except ImportError:
        logger.warning("matplotlib/seaborn未安装，跳过混淆矩阵图片保存")


def main():
    args = parse_args()
    
    # 设备
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if args.model_type == "domain":
        model = SciBERTDomainClassifier(model_name=args.model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        # 加载数据
        test_dataset = DomainDataset(args.test_data, max_length=args.max_length)
        test_loader = create_dataloaders(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        logger.info(f"测试集大小: {len(test_dataset)}")
        logger.info(f"标签分布: {test_dataset.get_label_distribution()}")
        
        # 评估
        metrics = evaluate_domain(model, test_loader, device)
        
        logger.info("\n" + "=" * 50)
        logger.info("Domain分类评估结果")
        logger.info("=" * 50)
        logger.info(format_metrics(metrics, ["NLP", "CV", "ML", "AI"]))
        
        # 保存结果
        result_path = os.path.join(args.output_dir, "domain_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, 
                     f, indent=2, ensure_ascii=False)
        
        # 保存混淆矩阵
        if args.save_confusion_matrix:
            cm_path = os.path.join(args.output_dir, "domain_confusion_matrix.png")
            save_confusion_matrix(np.array(metrics["confusion_matrix"]), ["NLP", "CV", "ML", "AI"], cm_path)
    
    elif args.model_type == "quality":
        model = SciBERTQualityClassifier(model_name=args.model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        test_dataset = QualityDataset(args.test_data, max_length=args.max_length)
        test_loader = create_dataloaders(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        logger.info(f"测试集大小: {len(test_dataset)}")
        logger.info(f"标签分布: {test_dataset.get_label_distribution()}")
        
        metrics = evaluate_quality(model, test_loader, device)
        
        logger.info("\n" + "=" * 50)
        logger.info("Quality分类评估结果")
        logger.info("=" * 50)
        logger.info(format_metrics(metrics, ["Acceptable", "Borderline", "Weak Reject"]))
        
        result_path = os.path.join(args.output_dir, "quality_results.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"},
                     f, indent=2, ensure_ascii=False)
        
        if args.save_confusion_matrix:
            cm_path = os.path.join(args.output_dir, "quality_confusion_matrix.png")
            save_confusion_matrix(np.array(metrics["confusion_matrix"]), ["Acceptable", "Borderline", "Weak Reject"], cm_path)
    
    elif args.model_type == "multitask":
        model = SciBERTMultiTaskClassifier(model_name=args.model_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        test_dataset = MultiTaskDataset(
            domain_data_path=args.domain_test or args.test_data,
            quality_data_path=args.quality_test or args.test_data,
            max_length=args.max_length,
            oversample_quality=False
        )
        test_loader = create_dataloaders(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        
        metrics = evaluate_multitask(model, test_loader, device)
        
        if "domain" in metrics:
            logger.info("\n" + "=" * 50)
            logger.info("Domain分类评估结果")
            logger.info("=" * 50)
            logger.info(format_metrics(metrics["domain"], ["NLP", "CV", "ML", "AI"]))
            
            result_path = os.path.join(args.output_dir, "domain_results.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in metrics["domain"].items() if k != "confusion_matrix"},
                         f, indent=2, ensure_ascii=False)
            
            if args.save_confusion_matrix:
                cm_path = os.path.join(args.output_dir, "domain_confusion_matrix.png")
                save_confusion_matrix(np.array(metrics["domain"]["confusion_matrix"]), 
                                     ["NLP", "CV", "ML", "AI"], cm_path)
        
        if "quality" in metrics:
            logger.info("\n" + "=" * 50)
            logger.info("Quality分类评估结果")
            logger.info("=" * 50)
            logger.info(format_metrics(metrics["quality"], ["Acceptable", "Borderline", "Weak Reject"]))
            
            result_path = os.path.join(args.output_dir, "quality_results.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in metrics["quality"].items() if k != "confusion_matrix"},
                         f, indent=2, ensure_ascii=False)
            
            if args.save_confusion_matrix:
                cm_path = os.path.join(args.output_dir, "quality_confusion_matrix.png")
                save_confusion_matrix(np.array(metrics["quality"]["confusion_matrix"]),
                                     ["Acceptable", "Borderline", "Weak Reject"], cm_path)
    
    logger.info(f"\n评估结果已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
