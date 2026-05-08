"""
Agent 编排层 Demo — 无需预训练模型即可运行的端到端演示

演示完整的 ReAct 多 Agent 协作审稿流程:
    1. Coordinator 任务分解
    2. ClassifierAgent (SciBERT zero-shot)
    3. ExtractorAgent (rule-based fallback)
    4. AnalystAgent (template analysis)
    5. ReviewerAgent (template review)
    6. 反思与完整性检查

Usage:
    python scripts/agent_demo.py
    python scripts/agent_demo.py --text "Title: Your Paper... Abstract: ..."
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.memory import WorkingMemory, KnowledgeGraph
from agents.specialized import ClassifierAgent, ExtractorAgent, AnalystAgent, ReviewerAgent
from agents.coordinator import CoordinatorAgent


class DemoClassifier:
    """Demo分类器 (SciBERT zero-shot)"""

    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
        self.model.eval()

        self.domain_keywords = {
            "NLP": ["language", "text", "translation", "token", "bert", "transformer", "sentence", "word", "corpus", "nlp", "parsing", "syntax"],
            "CV": ["image", "visual", "pixel", "convolution", "cnn", "resnet", "detection", "segmentation", "object", "video"],
            "ML": ["training", "optimization", "gradient", "loss", "parameter", "regularization", "bayesian", "kernel", "ensemble"],
            "AI": ["reasoning", "planning", "knowledge", "agent", "inference", "search", "logic", "representation"],
        }
        self.method_keywords = {
            "Empirical": ["experiment", "result", "benchmark", "dataset", "performance", "accuracy", "evaluate"],
            "Theoretical": ["theorem", "proof", "bound", "convergence", "lemma", "assumption", "complexity"],
            "Survey": ["survey", "review", "overview", "taxonomy", "literature", "comprehensive"],
            "Benchmark": ["benchmark", "baseline", "comparison", "standard", "leaderboard"],
        }

    def classify(self, text: str):
        text_lower = text.lower()
        # Domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            domain_scores[domain] = score
        domains = [d for d, s in domain_scores.items() if s > 0] or ["ML"]

        # Method
        method_scores = {}
        for method, keywords in self.method_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            method_scores[method] = score
        best_method = max(method_scores, key=method_scores.get) if max(method_scores.values()) > 0 else "Empirical"

        # Quality heuristic
        quality = "Acceptable"
        if len(text.split()) < 50:
            quality = "Borderline"

        return {
            "domains": domains[:3],
            "method_type": best_method,
            "quality_tier": quality,
            "confidence": {"domain": 0.8, "quality": 0.7, "method": 0.7}
        }


class DemoExtractor:
    """Demo信息抽取器 (正则 + 规则)"""

    def __init__(self):
        import re
        self.model_pattern = re.compile(
            r'\b(BERT|GPT-\d|RoBERTa|T5|BART|ResNet|VGG|Transformer|LSTM|CNN|RNN|'
            r'YOLO|ViT|CLIP|Diffusion|AlphaFold|DeepLab|U-Net)\b', re.IGNORECASE)
        self.dataset_pattern = re.compile(
            r'\b(SQuAD|ImageNet|COCO|GLUE|WMT|CIFAR|MNIST|WikiText|'
            r'ConceptNet|Freebase|PubMed|Arxiv|OpenSubtitles)\b', re.IGNORECASE)
        self.metric_pattern = re.compile(
            r'\b(F1|BLEU|ROUGE|Accuracy|Recall|Precision|AUC|MRR|NDCG|'
            r'Perplexity|MAP|IoU|mAP)\b', re.IGNORECASE)

    def extract_information(self, text: str):
        entities = []
        for m in self.model_pattern.finditer(text):
            entities.append({"text": m.group(), "type": "MODEL", "sent_id": 0})
        for m in self.dataset_pattern.finditer(text):
            entities.append({"text": m.group(), "type": "DATASET", "sent_id": 0})
        for m in self.metric_pattern.finditer(text):
            entities.append({"text": m.group(), "type": "METRIC", "sent_id": 0})
        return {"entities": entities, "triples": []}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Demo")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    demo_text = args.text or (
        "Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. "
        "Abstract: We introduce a new language representation model called BERT, which stands for "
        "Bidirectional Encoder Representations from Transformers. Unlike recent language representation "
        "models, BERT is designed to pre-train deep bidirectional representations from unlabeled text "
        "by jointly conditioning on both left and right context in all layers. As a result, the "
        "pre-trained BERT model can be fine-tuned with just one additional output layer to create "
        "state-of-the-art models for a wide range of tasks, such as question answering and language "
        "inference, without substantial task-specific architecture modifications. BERT advances the "
        "state of the art for eleven NLP tasks including pushing the GLUE score to 80.5%, MultiNLI "
        "accuracy to 86.7%, SQuAD v1.1 F1 to 93.2, and SQuAD v2.0 F1 to 83.1."
    )

    print("=" * 60)
    print("  基于多 Agent 协作的科研文献智能分析系统 - Demo")
    print("=" * 60)

    # 初始化组件
    device = args.device
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("\n初始化 NLP 模块...")
    classifier = DemoClassifier(device=device)
    extractor = DemoExtractor()
    from models.summarizer import ReviewGenerator
    review_gen = ReviewGenerator()

    print("\n创建 Agent 团队...")
    kg = KnowledgeGraph("knowledge_graph.json")
    coordinator = CoordinatorAgent(knowledge_graph=kg)

    coordinator.register_agent("ClassifierAgent", ClassifierAgent(classifier))
    coordinator.register_agent("ExtractorAgent", ExtractorAgent(extractor))
    coordinator.register_agent("AnalystAgent", AnalystAgent(summarizer=None))
    coordinator.register_agent("RetrieverAgent", None)  # 跳过检索
    coordinator.register_agent("ReviewerAgent", ReviewerAgent(review_generator=review_gen))

    print(f"\nAgent 团队: {[a for a, ag in coordinator.agents.items() if ag is not None]}")

    # 执行审稿
    print("\n" + "=" * 60)
    print("开始论文审稿...")
    print("=" * 60)

    memory = WorkingMemory(paper_id="demo_paper_001")
    memory.set("paper_text", demo_text)
    memory.set("paper_title", "BERT: Pre-training of Deep Bidirectional Transformers")

    final_report = coordinator.run(memory)

    # 输出报告
    print("\n" + "=" * 60)
    print("  最终审稿报告")
    print("=" * 60)

    ps = final_report.get("paper_summary", {})
    print(f"\n论文分类:")
    print(f"  领域: {ps.get('domains', [])}")
    print(f"  方法类型: {ps.get('method_type', '?')}")
    print(f"  质量评估: {ps.get('quality_tier', '?')}")

    es = final_report.get("extraction_stats", {})
    print(f"\n信息抽取统计:")
    print(f"  实体数量: {es.get('entities', 0)}")
    print(f"  实体类型: {es.get('entity_types', {})}")
    print(f"  三元组数: {es.get('triples', 0)}")

    ea = final_report.get("experiment_analysis", {})
    print(f"\n实验分析:")
    print(f"  数据集: {ea.get('datasets', [])}")
    print(f"  指标: {ea.get('metrics', [])}")
    print(f"  优势: {ea.get('strengths', [])}")
    print(f"  关注点: {ea.get('concerns', [])}")

    rd = final_report.get("review_draft", {})
    print(f"\n审稿意见:")
    print(f"  优势: {rd.get('strengths', [])}")
    print(f"  不足: {rd.get('weaknesses', [])}")
    print(f"  建议: {rd.get('suggestions', [])}")
    print(f"  可复现性: {rd.get('reproducibility_notes', [])}")

    oa = rd.get("overall_assessment", final_report.get("overall_assessment", ""))
    print(f"\n整体评估:")
    print(f"  {oa}")

    rn = final_report.get("reflection_notes", [])
    if rn:
        print(f"\n反思备注:")
        for note in rn:
            print(f"  - {note}")

    print(f"\nReAct 推理步骤: {len(coordinator.trace)} 步")
    for i, step in enumerate(coordinator.trace):
        print(f"  [{i+1}] {step.thought[:80]}...")

    print("\nDemo 完成! 各模块可替换为微调模型以提升效果。")


if __name__ == "__main__":
    main()
