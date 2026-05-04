"""
分类器推理脚本

提供两种使用方式:
    1. 命令行: 输入论文文本或JSON文件，输出分类结果
    2. Python API: 供Agent层直接调用 classify_paper()

Usage:
    # 单篇论文分类
    python -m train.classifier.predict \
        --model_path checkpoints/domain_merged/best_model.pt \
        --model_type domain \
        --text "Title: BERT: Pre-training... Abstract: We introduce..."

    # 批量分类 (JSON文件)
    python -m train.classifier.predict \
        --model_path checkpoints/multitask/best_model.pt \
        --model_type multitask \
        --input_file papers.json \
        --output_file predictions.json

    # papers.json 格式:
    # [
    #   {"text": "Title: ... Abstract: ..."},
    #   {"text": "Title: ... Abstract: ..."}
    # ]
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from models.classifier.scibert_classifier import PaperClassifier, classify_paper


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="分类器推理")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["domain", "quality", "multitask"],
                        help="模型类型")
    
    # 输入 (二选一)
    parser.add_argument("--text", type=str, default=None,
                        help="单篇论文文本 (Title: ... Abstract: ...)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="批量输入JSON文件路径")
    
    # 输出
    parser.add_argument("--output_file", type=str, default=None,
                        help="批量输出JSON文件路径")
    
    # 设备
    parser.add_argument("--device", type=str, default=None)
    
    return parser.parse_args()


def predict_single(classifier: PaperClassifier, text: str) -> Dict:
    """
    对单篇论文进行分类
    
    Args:
        classifier: PaperClassifier实例
        text: 论文文本
    
    Returns:
        分类结果字典
    """
    result = classifier.classify(text)
    result["input_preview"] = text[:100] + "..." if len(text) > 100 else text
    return result


def predict_batch(classifier: PaperClassifier, texts: List[str], batch_size: int = 16) -> List[Dict]:
    """
    批量分类
    
    Args:
        classifier: PaperClassifier实例
        texts: 论文文本列表
        batch_size: 批大小
    
    Returns:
        分类结果列表
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_results = classifier.classify_batch(batch_texts)
        
        for text, result in zip(batch_texts, batch_results):
            result["input_preview"] = text[:100] + "..." if len(text) > 100 else text
            results.append(result)
    
    return results


def main():
    args = parse_args()
    
    # 检查输入
    if args.text is None and args.input_file is None:
        logger.error("请提供 --text 或 --input_file 参数")
        return
    
    # 设备
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    classifier = PaperClassifier(
        model_path=args.model_path,
        model_type=args.model_type,
        device=device
    )
    
    if args.text:
        # 单篇分类
        logger.info("=" * 50)
        logger.info("单篇论文分类")
        logger.info("=" * 50)
        
        result = predict_single(classifier, args.text)
        
        logger.info("\n分类结果:")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 保存结果
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存到 {args.output_file}")
    
    elif args.input_file:
        # 批量分类
        logger.info("=" * 50)
        logger.info("批量分类")
        logger.info("=" * 50)
        
        # 读取输入文件
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        
        if isinstance(input_data, list):
            texts = [item.get("text", "") for item in input_data if item.get("text")]
        elif isinstance(input_data, dict):
            texts = [input_data.get("text", "")]
        else:
            logger.error("输入文件格式错误，应为JSON对象或对象列表")
            return
        
        logger.info(f"共 {len(texts)} 篇论文待分类")
        
        # 批量推理
        results = predict_batch(classifier, texts)
        
        # 统计
        domain_dist = {}
        quality_dist = {}
        for r in results:
            for d in r.get("domains", []):
                domain_dist[d] = domain_dist.get(d, 0) + 1
            q = r.get("quality_tier", "Unknown")
            quality_dist[q] = quality_dist.get(q, 0) + 1
        
        logger.info("\n领域分布:")
        for domain, count in sorted(domain_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {domain}: {count} ({count/len(results)*100:.1f}%)")
        
        logger.info("\n质量分布:")
        for quality, count in sorted(quality_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {quality}: {count} ({count/len(results)*100:.1f}%)")
        
        # 保存结果
        output_file = args.output_file or args.input_file.replace(".json", "_predictions.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
