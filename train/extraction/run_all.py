"""
信息抽取训练一键入口

根据配置文件自动训练所有抽取模型:
    1. NER模型 (SciBERT + BiLSTM + CRF)
    2. RE模型 (SciBERT + 实体标记 + 关系分类)

Usage:
    python -m train.extraction.run_all --processed_data_dir processed_data --output_dir checkpoints
    python -m train.extraction.run_all --models ner re
    python -m train.extraction.run_all --batch_size 16 --lr 5e-5 --epochs 10
"""

import os
import sys
import argparse
import logging
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="一键训练所有信息抽取模型")

    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    parser.add_argument("--models", type=str, nargs="+",
                        choices=["ner", "re", "all"], default=["all"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def run_command(cmd, description):
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


def train_ner(args):
    data_dir = os.path.join(args.processed_data_dir, "scierc_ner_re")
    output_dir = os.path.join(args.output_dir, "ner")

    train_file = os.path.join(data_dir, "ner_train.jsonl")
    dev_file = os.path.join(data_dir, "ner_dev.jsonl")
    test_file = os.path.join(data_dir, "ner_test.jsonl")

    if not os.path.exists(train_file):
        logger.warning(f"NER训练数据不存在: {train_file}，跳过")
        logger.info("请先运行: python scripts/scierc_processed.py")
        return False

    cmd = [
        sys.executable, "-m", "train.extraction.train_ner",
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
    return run_command(cmd, "NER模型训练")


def train_re(args):
    data_dir = os.path.join(args.processed_data_dir, "scierc_ner_re")
    output_dir = os.path.join(args.output_dir, "re")

    train_file = os.path.join(data_dir, "re_train.jsonl")
    dev_file = os.path.join(data_dir, "re_dev.jsonl")
    test_file = os.path.join(data_dir, "re_test.jsonl")

    if not os.path.exists(train_file):
        logger.warning(f"RE训练数据不存在: {train_file}，跳过")
        logger.info("请先运行: python scripts/scierc_processed.py")
        return False

    cmd = [
        sys.executable, "-m", "train.extraction.train_re",
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
    return run_command(cmd, "RE模型训练")


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("信息抽取训练一键入口")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.processed_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"训练模型: {args.models}")

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["ner", "re"]

    results = {}

    if "ner" in models_to_train:
        results["ner"] = train_ner(args)

    if "re" in models_to_train:
        results["re"] = train_re(args)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成总结")
    logger.info("=" * 60)
    for model_name, success in results.items():
        status = "成功" if success else "失败/跳过"
        logger.info(f"  {model_name}: {status}")

    logger.info(f"\n模型检查点保存在: {args.output_dir}")
    logger.info("  - NER: checkpoints/ner/best_model.pt")
    logger.info("  - RE: checkpoints/re/best_model.pt")


if __name__ == "__main__":
    main()
