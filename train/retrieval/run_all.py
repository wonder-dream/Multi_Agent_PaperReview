"""
检索系统一键构建入口

步骤:
    1. 构建 FAISS 论文语义索引
    2. (可选) 运行评估验证检索质量

Usage:
    python -m train.retrieval.run_all --output_dir checkpoints/retrieval
    python -m train.retrieval.run_all --encoder_model allenai/scibert_scivocab_uncased --max_papers 2000
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
    parser = argparse.ArgumentParser(description="一键构建语义检索系统")
    parser.add_argument("--processed_data_dir", type=str, default="processed_data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/retrieval")
    parser.add_argument("--encoder_model", type=str, default="allenai/specter")
    parser.add_argument("--index_type", type=str, default="flat_ip",
                        choices=["flat_ip", "hnsw"])
    parser.add_argument("--max_papers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def run_command(cmd, description):
    logger.info(f"\n{'='*60}")
    logger.info(f"开始: {description}")
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
    logger.info("语义检索系统一键构建")
    logger.info("=" * 60)
    logger.info(f"编码器: {args.encoder_model}")
    logger.info(f"索引类型: {args.index_type}")
    logger.info(f"输出目录: {args.output_dir}")

    # Step 1: 构建索引
    cmd = [
        sys.executable, "-m", "train.retrieval.build_index",
        "--processed_data_dir", args.processed_data_dir,
        "--output_dir", args.output_dir,
        "--encoder_model", args.encoder_model,
        "--index_type", args.index_type,
        "--batch_size", str(args.batch_size),
    ]
    if args.max_papers > 0:
        cmd.extend(["--max_papers", str(args.max_papers)])
    if args.device:
        cmd.extend(["--device", args.device])

    success = run_command(cmd, "构建 FAISS 语义索引")

    if success and not args.skip_eval:
        logger.info("\n索引构建成功，可运行以下命令评估检索质量:")
        logger.info(f"  python -m train.retrieval.evaluate "
                    f"--index_dir {args.output_dir} "
                    f"--output_dir results/retrieval")

    logger.info(f"\n索引位置: {args.output_dir}")
    logger.info("Agent 加载方式:")
    logger.info(f"  from models.retrieval import PaperRetriever")
    logger.info(f"  retriever = PaperRetriever(index_dir='{args.output_dir}')")
    logger.info(f"  results = retriever.semantic_search('Title: ... Abstract: ...', top_k=5)")


if __name__ == "__main__":
    main()
