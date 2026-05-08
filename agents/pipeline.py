"""
Agent 编排层 CLI 入口 — 端到端论文审稿

Usage:
    # 基本用法 (仅需分类器，其他模块可选)
    python -m agents.pipeline \
        --text "Title: BERT: Pre-training of Deep Bidirectional Transformers... Abstract: We introduce..."

    # 从文件读取论文
    python -m agents.pipeline --input_file paper.json --output_file review_report.json

    # 完整配置
    python -m agents.pipeline \
        --text "..." \
        --classifier_model checkpoints/multitask/best_model.pt \
        --ner_model checkpoints/ner/best_model.pt \
        --re_model checkpoints/re/best_model.pt \
        --retrieval_index checkpoints/retrieval \
        --output_file review_report.json
"""
import os
import sys
import json
import argparse
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.orchestrator import PaperReviewOrchestrator, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="多Agent论文审稿系统")

    # 输入
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--paper_title", type=str, default="")

    # 模块路径
    parser.add_argument("--classifier_model", type=str, default="checkpoints/multitask/best_model.pt")
    parser.add_argument("--classifier_type", type=str, default="multitask")
    parser.add_argument("--ner_model", type=str, default="checkpoints/ner/best_model.pt")
    parser.add_argument("--re_model", type=str, default="checkpoints/re/best_model.pt")
    parser.add_argument("--generative_model", type=str, default=None)
    parser.add_argument("--retrieval_index", type=str, default=None)

    # 输出
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="review_results")
    parser.add_argument("--save_knowledge_graph", action="store_true")

    # 设备
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.text is None and args.input_file is None:
        logger.error("请提供 --text 或 --input_file")
        return

    # 读取输入
    papers = []
    if args.text:
        papers.append({"text": args.text, "title": args.paper_title, "id": "cli_input"})
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            papers = [{"text": d.get("text", ""), "title": d.get("title", ""),
                       "id": d.get("id", str(i))} for i, d in enumerate(data) if d.get("text")]
        elif isinstance(data, dict) and data.get("text"):
            papers = [{"text": data["text"], "title": data.get("title", ""),
                       "id": data.get("id", "cli_input")}]

    if not papers:
        logger.error("无有效论文输入")
        return

    # 构建配置
    config = PipelineConfig(
        classifier_model_path=args.classifier_model,
        classifier_model_type=args.classifier_type,
        ner_model_path=args.ner_model,
        re_model_path=args.re_model,
        generative_model_path=args.generative_model,
        retrieval_index_dir=args.retrieval_index,
        device=args.device
    )

    # 初始化编排器
    orchestrator = PaperReviewOrchestrator(config)
    orchestrator.initialize()

    # 执行审稿
    all_reports = []
    for i, paper in enumerate(papers):
        print(f"\n{'#'*60}")
        print(f"审稿进度: {i+1}/{len(papers)}")
        print(f"{'#'*60}")

        report = orchestrator.run_review(
            paper_text=paper["text"],
            paper_id=paper.get("id", f"paper_{i}"),
            paper_title=paper.get("title", "")
        )
        all_reports.append(report)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    if len(all_reports) == 1 and args.output_file:
        output_path = args.output_file
    else:
        output_path = os.path.join(args.output_dir, "review_reports.json")

    orchestrator.save_report(
        all_reports[0] if len(all_reports) == 1 else all_reports,
        output_path
    )

    # 保存知识图谱
    if args.save_knowledge_graph:
        orchestrator.save_knowledge_graph()

    # 打印摘要
    for report in all_reports:
        meta = report.get("meta", {})
        print(f"\n审稿摘要 [{meta.get('paper_id', '?')}]:")
        print(f"  领域: {report.get('paper_summary', {}).get('domains', [])}")
        print(f"  质量: {report.get('paper_summary', {}).get('quality_tier', '?')}")
        print(f"  实体: {report.get('extraction_stats', {}).get('entities', 0)}")
        print(f"  审稿用时: {meta.get('review_time', 0):.1f}s")
        od = report.get("review_draft", {})
        if od.get("overall_assessment"):
            print(f"  整体评估: {od['overall_assessment'][:120]}...")


if __name__ == "__main__":
    main()
