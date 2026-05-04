"""
分类器评估指标

提供:
    - 分类常用指标: Accuracy, Macro-F1, Weighted-F1, Per-class F1
    - 混淆矩阵计算
    - 分类报告生成
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_names: List[str] = None
) -> Dict:
    """
    计算分类任务的完整评估指标
    
    Args:
        predictions: 预测标签, shape (N,)
        labels: 真实标签, shape (N,)
        num_classes: 类别数量
        class_names: 类别名称列表
    
    Returns:
        指标字典:
            - accuracy: 准确率
            - macro_f1: Macro-F1 (关注少数类)
            - weighted_f1: Weighted-F1
            - macro_precision: Macro精确率
            - macro_recall: Macro召回率
            - per_class_f1: 每个类的F1分数
            - confusion_matrix: 混淆矩阵
    """
    metrics = {}
    
    # 基础指标
    metrics["accuracy"] = float(accuracy_score(labels, predictions))
    metrics["macro_f1"] = float(f1_score(labels, predictions, average="macro", zero_division=0))
    metrics["weighted_f1"] = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    metrics["macro_precision"] = float(precision_score(labels, predictions, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(labels, predictions, average="macro", zero_division=0))
    
    # 每个类的F1
    per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    metrics["per_class_f1"] = {
        name: float(score) 
        for name, score in zip(class_names, per_class_f1)
    }
    
    # 混淆矩阵
    metrics["confusion_matrix"] = confusion_matrix(labels, predictions, labels=list(range(num_classes))).tolist()

    return metrics


def compute_multilabel_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5
) -> Dict:
    """
    计算多标签分类指标

    Args:
        predictions: 预测概率, shape (N, num_classes)
        labels: 多热真实标签, shape (N, num_classes)
        class_names: 类别名称
        threshold: 二值化阈值

    Returns:
        指标字典
    """
    preds_binary = (predictions >= threshold).astype(int)

    per_class_f1 = f1_score(labels, preds_binary, average=None, zero_division=0)
    macro_f1 = float(f1_score(labels, preds_binary, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels, preds_binary, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds_binary, average="weighted", zero_division=0))
    # 子集准确率 (exact match ratio)
    subset_acc = float(np.mean((preds_binary == labels).all(axis=1)))

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "subset_accuracy": subset_acc,
        "per_class_f1": {name: float(score) for name, score in zip(class_names, per_class_f1)},
    }


def format_metrics(metrics: Dict, class_names: List[str] = None) -> str:
    """
    格式化指标输出为可读字符串
    
    Args:
        metrics: compute_classification_metrics的输出
        class_names: 类别名称
    
    Returns:
        格式化字符串
    """
    lines = [
        "=" * 50,
        "Classification Metrics",
        "=" * 50,
        f"Accuracy:        {metrics['accuracy']:.4f}",
        f"Macro-F1:        {metrics['macro_f1']:.4f}",
        f"Weighted-F1:     {metrics['weighted_f1']:.4f}",
        f"Macro-Precision: {metrics['macro_precision']:.4f}",
        f"Macro-Recall:    {metrics['macro_recall']:.4f}",
        "-" * 50,
        "Per-Class F1:"
    ]
    
    for name, score in metrics["per_class_f1"].items():
        lines.append(f"  {name:15s} {score:.4f}")
    
    lines.append("=" * 50)
    return "\n".join(lines)


def compute_metrics_from_outputs(outputs_list: List[Dict], task: str = "domain") -> Dict:
    """
    从模型输出列表计算指标 (用于训练循环中的评估)
    
    Args:
        outputs_list: 模型输出列表，每个元素是forward的返回字典
        task: 'domain' 或 'quality'
    
    Returns:
        指标字典
    """
    all_preds = []
    all_labels = []
    
    for output in outputs_list:
        if task == "domain":
            logits = output["logits"]
            labels = output.get("labels", None)
        elif task == "quality":
            logits = output["logits"]
            labels = output.get("labels", None)
        else:
            continue
        
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())
    
    if not all_labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if task == "domain":
        class_names = ["NLP", "CV", "ML", "AI"]
        return compute_classification_metrics(all_preds, all_labels, 4, class_names)
    else:
        class_names = ["Acceptable", "Borderline", "Weak Reject"]
        return compute_classification_metrics(all_preds, all_labels, 2, class_names)


def compute_multitask_metrics(outputs_list: List[Dict]) -> Dict:
    """
    计算多任务模型的指标
    
    Args:
        outputs_list: 多任务模型输出列表
    
    Returns:
        包含domain和quality指标的字典
    """
    domain_preds = []
    domain_labels = []
    quality_preds = []
    quality_labels = []
    
    for output in outputs_list:
        if "domain_logits" in output and "domain_labels" in output:
            preds = torch.argmax(output["domain_logits"], dim=-1).cpu().numpy()
            labels = output["domain_labels"].cpu().numpy()
            valid = labels >= 0
            domain_preds.extend(preds[valid].tolist())
            domain_labels.extend(labels[valid].tolist())

        if "quality_logits" in output and "quality_labels" in output:
            preds = torch.argmax(output["quality_logits"], dim=-1).cpu().numpy()
            labels = output["quality_labels"].cpu().numpy()
            valid = labels >= 0
            quality_preds.extend(preds[valid].tolist())
            quality_labels.extend(labels[valid].tolist())
    
    result = {}
    
    if domain_labels:
        result["domain"] = compute_classification_metrics(
            np.array(domain_preds), 
            np.array(domain_labels), 
            4, 
            ["NLP", "CV", "ML", "AI"]
        )
    
    if quality_labels:
        result["quality"] = compute_classification_metrics(
            np.array(quality_preds),
            np.array(quality_labels),
            2,
            ["accept", "reject"]
        )
    
    return result
