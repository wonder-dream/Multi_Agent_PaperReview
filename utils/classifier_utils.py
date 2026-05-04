"""
分类器工具函数

提供训练过程中常用的辅助功能:
    - 随机种子设置
    - 类别权重计算
    - 配置保存/加载
    - 日志设置
    - 学习率调度器创建
"""

import os
import json
import random
import logging
from typing import Dict, Optional

import torch
import numpy as np


def set_seed(seed: int = 42):
    """
    设置全局随机种子，保证实验可复现
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保CUDA的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"[Utils] 随机种子已设置为 {seed}")


def compute_class_weights(label_counts: list, method: str = "inverse") -> torch.Tensor:
    """
    计算类别权重 (用于处理类别不平衡)
    
    Args:
        label_counts: 每个类别的样本数列表 [count_0, count_1, ...]
        method: 权重计算方法
            - "inverse": 1 / count
            - "inverse_sqrt": 1 / sqrt(count)
            - "effective": 有效样本数加权 (1-beta)/(1-beta^count)
    
    Returns:
        权重张量，shape (num_classes,)
    
    Example:
        >>> weights = compute_class_weights([8000, 2000], method="inverse")
        >>> print(weights)  # tensor([0.25, 1.0])
    """
    label_counts = np.array(label_counts, dtype=np.float32)
    
    if method == "inverse":
        weights = 1.0 / label_counts
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(label_counts)
    elif method == "effective":
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** label_counts)
    else:
        raise ValueError(f"Unknown weight method: {method}")
    
    # 归一化
    weights = weights / weights.sum() * len(weights)
    
    return torch.tensor(weights, dtype=torch.float)


def save_config(config: Dict, save_path: str):
    """
    保存训练配置到JSON文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logging.info(f"[Utils] 配置已保存到 {save_path}")


def load_config(config_path: str) -> Dict:
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(levelname)s - %(message)s"
):
    """
    设置日志
    
    Args:
        log_file: 日志文件路径 (None则只输出到控制台)
        level: 日志级别
        format_str: 日志格式
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True
    )


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        参数字典:
            - total: 总参数量
            - trainable: 可训练参数量
            - frozen: 冻结参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen
    }


def print_model_info(model: torch.nn.Module):
    """
    打印模型信息 (参数量、层数等)
    
    Args:
        model: PyTorch模型
    """
    params = count_parameters(model)
    total_m = params["total"] / 1e6
    trainable_m = params["trainable"] / 1e6
    frozen_m = params["frozen"] / 1e6
    
    logging.info("=" * 50)
    logging.info("模型信息")
    logging.info("=" * 50)
    logging.info(f"总参数量:     {params['total']:,} ({total_m:.2f}M)")
    logging.info(f"可训练参数:   {params['trainable']:,} ({trainable_m:.2f}M)")
    logging.info(f"冻结参数:     {params['frozen']:,} ({frozen_m:.2f}M)")
    logging.info("=" * 50)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        device_str: 设备字符串 (None/'auto' 自动选择)
    
    Returns:
        torch.device实例
    """
    if device_str is None or device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    logging.info(f"[Utils] 使用设备: {device}")
    if device.type == "cuda":
        logging.info(f"[Utils] GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"[Utils] 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return device


def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读时间字符串
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化字符串 (如 "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    save_path: str,
    scheduler=None
):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 指标字典
        save_path: 保存路径
        scheduler: 学习率调度器 (可选)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    logging.info(f"[Utils] 检查点已保存: {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None
) -> Dict:
    """
    加载训练检查点
    
    Args:
        model: 模型 (会就地加载权重)
        checkpoint_path: 检查点路径
        optimizer: 优化器 (可选，会就地加载状态)
        scheduler: 学习率调度器 (可选)
        device: 设备
    
    Returns:
        检查点中的其他信息 (epoch, metrics等)
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logging.info(f"[Utils] 检查点已加载: {checkpoint_path} (Epoch {checkpoint.get('epoch', '?')})")
    
    return {k: v for k, v in checkpoint.items() 
            if k not in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]}
