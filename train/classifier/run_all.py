"""
分类器训练一键入口

根据配置文件自动训练所有分类器:
    1. Domain分类器 (arXiv + PeerRead合并数据)
    2. Quality分类器 (PeerRead数据，处理类别不平衡)
    3. MultiTask分类器 (联合训练)

Usage:
    # 使用默认配置训练全部
    python -m train.classifier.run_all --processed_data_dir processed_data --output_dir checkpoints

    # 只训练指定模型
    python -m train.classifier.run_all --models domain quality

    # 指定超参数
    python -m train.classifier.run_all --batch_size 32 --lr 2e-5 --epochs 5
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="一键训练所有分类器")
    
    # 数据目录
    parser.add_argument("--processed_data_dir", type=str, default="processed_data",
                        help="处理后数据的根目录")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="模型输出根目录")
    
    # 选择要训练的模型
    parser.add_argument("--models", type=str, nargs="+",
                        choices=["domain", "quality", "multitask", "all"],
                        default=["all"],
                        help="要训练的模型")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 设备
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def run_command(cmd: list, description: str):
    """运行子进程命令"""
    logger.info(f"\n{'='*60}")
    logger.info(f"开始: {description}")
    logger.info(f"命令: {' '.join(cmd)}")
    logger.info(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        logger.error(f"{description} 失败!")
        return False
    
    logger.info(f"{description} 完成!")
    return True


def train_domain(args: argparse.Namespace) -> bool:
    """训练Domain分类器"""
    data_dir = os.path.join(args.processed_data_dir, "arxiv_PeerRead_merge_data", "classification")
    output_dir = os.path.join(args.output_dir, "domain")
    
    # 检查数据文件
    train_file = os.path.join(data_dir, "merged_train.jsonl")
    dev_file = os.path.join(data_dir, "merged_dev.jsonl")
    test_file = os.path.join(data_dir, "merged_test.jsonl")
    
    if not os.path.exists(train_file):
        logger.warning(f"Domain训练数据不存在: {train_file}，跳过")
        return False
    
    cmd = [
        sys.executable, "-m", "train.classifier.train_domain",
        "--train_data", train_file,
        "--dev_data", dev_file,
        "--test_data", test_file,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--max_length", str(args.max_length),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers)
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    return run_command(cmd, "Domain分类器训练")


def train_quality(args: argparse.Namespace) -> bool:
    """训练Quality分类器"""
    data_dir = os.path.join(args.processed_data_dir, "arxiv_PeerRead_merge_data", "classification")
    output_dir = os.path.join(args.output_dir, "quality")
    
    train_file = os.path.join(data_dir, "quality_train.jsonl")
    dev_file = os.path.join(data_dir, "quality_dev.jsonl")
    test_file = os.path.join(data_dir, "quality_test.jsonl")
    
    if not os.path.exists(train_file):
        logger.warning(f"Quality训练数据不存在: {train_file}，跳过")
        return False
    
    cmd = [
        sys.executable, "-m", "train.classifier.train_quality",
        "--train_data", train_file,
        "--dev_data", dev_file,
        "--test_data", test_file,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--max_length", str(args.max_length),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--use_class_weights"
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    return run_command(cmd, "Quality分类器训练")


def train_multitask(args: argparse.Namespace) -> bool:
    """训练MultiTask分类器"""
    data_dir = os.path.join(args.processed_data_dir, "arxiv_PeerRead_merge_data", "classification")
    output_dir = os.path.join(args.output_dir, "multitask")
    
    domain_train = os.path.join(data_dir, "merged_train.jsonl")
    domain_dev = os.path.join(data_dir, "merged_dev.jsonl")
    domain_test = os.path.join(data_dir, "merged_test.jsonl")
    
    quality_train = os.path.join(data_dir, "quality_train.jsonl")
    quality_dev = os.path.join(data_dir, "quality_dev.jsonl")
    quality_test = os.path.join(data_dir, "quality_test.jsonl")
    
    if not all(os.path.exists(f) for f in [domain_train, quality_train]):
        logger.warning("MultiTask训练数据不完整，跳过")
        return False
    
    cmd = [
        sys.executable, "-m", "train.classifier.train_multitask",
        "--domain_train", domain_train,
        "--domain_dev", domain_dev,
        "--domain_test", domain_test,
        "--quality_train", quality_train,
        "--quality_dev", quality_dev,
        "--quality_test", quality_test,
        "--output_dir", output_dir,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--max_length", str(args.max_length),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--oversample_quality"
    ]
    
    if args.device:
        cmd.extend(["--device", args.device])
    
    return run_command(cmd, "MultiTask联合训练")


def main():
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("分类器训练一键入口")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.processed_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练模型: {args.models}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["domain", "quality", "multitask"]
    
    results = {}
    
    if "domain" in models_to_train:
        results["domain"] = train_domain(args)
    
    if "quality" in models_to_train:
        results["quality"] = train_quality(args)
    
    if "multitask" in models_to_train:
        results["multitask"] = train_multitask(args)
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("训练完成总结")
    logger.info("=" * 60)
    for model_name, success in results.items():
        status = "成功" if success else "失败/跳过"
        logger.info(f"  {model_name}: {status}")
    
    logger.info(f"\n模型检查点保存在: {args.output_dir}")
    logger.info("  - Domain: checkpoints/domain/best_model.pt")
    logger.info("  - Quality: checkpoints/quality/best_model.pt")
    logger.info("  - MultiTask: checkpoints/multitask/best_model.pt")


if __name__ == "__main__":
    main()
