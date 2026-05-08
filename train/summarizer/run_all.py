"""
摘要生成训练一键入口

根据配置文件训练摘要生成模型:
    1. BART 摘要模型 (SciTLDR 微调)
    2. 可选: 冻结 encoder/decoder 加速训练

Usage:
    python -m train.summarizer.run_all --processed_data_dir processed_data --output_dir checkpoints
    python -m train.summarizer.run_all --model_name facebook/bart-base --batch_size 4 --epochs 5
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
    parser = argparse.ArgumentParser(description="一键训练摘要生成模型")

    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    parser.add_argument("--model_name", type=str, default="facebook/bart-base",
                        help="BART 模型名称 (base/large)")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)

    parser.add_argument("--freeze_encoder", type=int, default=0)
    parser.add_argument("--freeze_decoder", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
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


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("摘要生成训练一键入口")
    logger.info("=" * 60)
    logger.info(f"数据目录: {args.processed_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"模型: {args.model_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    data_dir = os.path.join(args.processed_data_dir, "sciTLDR_data")
    output_dir = os.path.join(args.output_dir, "summarizer")

    train_file = os.path.join(data_dir, "train.jsonl")
    dev_file = os.path.join(data_dir, "dev.jsonl")
    test_file = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_file):
        logger.warning(f"训练数据不存在: {train_file}，跳过")
        return

    cmd = [
        sys.executable, "-m", "train.summarizer.train_summarizer",
        "--train_data", train_file,
        "--dev_data", dev_file,
        "--test_data", test_file,
        "--output_dir", output_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--max_source_length", str(args.max_source_length),
        "--max_target_length", str(args.max_target_length),
        "--patience", str(args.patience),
        "--seed", str(args.seed),
        "--num_workers", str(args.num_workers),
        "--freeze_encoder", str(args.freeze_encoder),
        "--freeze_decoder", str(args.freeze_decoder),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    success = run_command(cmd, "BART 摘要模型训练")

    logger.info(f"\n模型检查点: {output_dir}/best_model.pt")
    status = "成功" if success else "失败"
    logger.info(f"训练状态: {status}")


if __name__ == "__main__":
    main()
