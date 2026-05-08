"""
审稿意见生成器: 基于模板 + 实体/关系的提示性审稿意见

模板策略:
    1. 基于抽取的实体和关系生成结构化的审稿提示
    2. 覆盖: 优势/不足/实验/方法/对比 等多个审稿维度
    3. 可配合 BART 生成更自然的审稿文本

审稿维度:
    - 贡献总结
    - 方法评估
    - 实验完整性
    - 对比公正性
    - 可复现性
"""
from typing import Dict, List
import random

random.seed(42)

REVIEW_TEMPLATES = {
    "strength": [
        "论文在 {dataset} 上的 {metric} 表现突出，达到了 {value}。",
        "提出的 {method} 方法在 {task} 任务上取得了显著提升。",
        "实验设计严谨，在 {dataset} 等多个数据集上进行了验证。",
        "论文对 {method} 的创新点阐述清晰，理论支撑充分。",
    ],
    "weakness": [
        "实验仅在 {dataset} 上进行，缺少在 {other_dataset} 等更多数据集上的跨域验证。",
        "方法使用了 {method}，但未与 {baseline} 进行充分的公平对比。",
        "缺少对 {missing_item} 的消融实验分析，难以评估各模块的独立贡献。",
        "论文未讨论 {limitation} 等潜在局限性。",
        "超参数 {hyperparam} 的选择缺乏充分的理论依据或敏感性分析。",
    ],
    "suggestion": [
        "建议在 {dataset_2} 等数据集上补充实验，以验证方法的泛化能力。",
        "建议增加与 {model} 的对比实验，作为更强的 baseline。",
        "建议补充 {analysis} 分析，增强实验结论的说服力。",
        "建议在论文中明确讨论 {concern} 的局限性及未来改进方向。",
    ],
    "reproducibility": [
        "论文是否公开了代码和数据？若未公开，建议补充。",
        "实验超参数设置描述清晰，具备较好的可复现性。",
        "伪代码与实际实现之间可能存在差异，建议提供可运行代码。",
    ]
}

DEFAULT_DATASETS = ["SQuAD", "ImageNet", "COCO", "WMT14", "GLUE", "CIFAR-10"]
DEFAULT_METRICS = ["F1", "BLEU", "Accuracy", "ROUGE", "MRR", "NDCG"]
DEFAULT_METHODS = ["fine-tuning", "pre-training", "data augmentation", "ensemble"]
DEFAULT_TASKS = ["classification", "NER", "QA", "translation", "summarization"]
DEFAULT_MODELS = ["BERT", "GPT", "T5", "ResNet", "LSTM", "Transformer"]


class ReviewGenerator:
    """审稿意见生成器"""

    def __init__(self, use_extraction: bool = True):
        self.use_extraction = use_extraction

    def generate(self, entities: List[Dict], triples: List[Dict],
                 paper_info: Dict = None) -> Dict:
        """
        生成结构化审稿意见

        Args:
            entities: NER抽取的实体列表 [{"text": "BERT", "type": "MODEL"}, ...]
            triples: RE抽取的三元组 [{"head": "BERT", "relation": "...", "tail": "..."}, ...]
            paper_info: 论文基本信息 (领域、质量等)

        Returns:
            {
                "strengths": [...],
                "weaknesses": [...],
                "suggestions": [...],
                "reproducibility_notes": [...],
                "overall_assessment": "..."
            }
        """
        # 从抽取结果中提取信息
        datasets = [e["text"] for e in entities if e["type"] == "DATASET"]
        metrics = [e["text"] for e in entities if e["type"] == "METRIC"]
        methods = [e["text"] for e in entities if e["type"] == "METHOD"]
        tasks = [e["text"] for e in entities if e["type"] == "TASK"]
        models = [e["text"] for e in entities if e.get("type") in ("MODEL", "GENERIC")]

        context = {
            "dataset": datasets[0] if datasets else DEFAULT_DATASETS[0],
            "dataset_2": random.choice([d for d in DEFAULT_DATASETS if d not in datasets] or DEFAULT_DATASETS),
            "other_dataset": random.choice([d for d in DEFAULT_DATASETS if d not in datasets] or DEFAULT_DATASETS),
            "metric": metrics[0] if metrics else DEFAULT_METRICS[0],
            "value": "state-of-the-art",
            "method": methods[0] if methods else DEFAULT_METHODS[0],
            "task": tasks[0] if tasks else DEFAULT_TASKS[0],
            "model": models[0] if models else DEFAULT_MODELS[0],
            "baseline": random.choice([m for m in DEFAULT_MODELS if m not in models] or DEFAULT_MODELS),
            "missing_item": "ablation study",
            "limitation": "模型复杂度与推理效率",
            "hyperparam": "learning rate",
            "analysis": "error analysis",
            "concern": "模型过拟合风险",
        }

        strengths = self._fill_templates(REVIEW_TEMPLATES["strength"], context, n=2)
        weaknesses = self._fill_templates(REVIEW_TEMPLATES["weakness"], context, n=3)
        suggestions = self._fill_templates(REVIEW_TEMPLATES["suggestion"], context, n=2)
        repro_notes = random.sample(REVIEW_TEMPLATES["reproducibility"], min(2, len(REVIEW_TEMPLATES["reproducibility"])))

        overall = self._generate_overall(paper_info or {}, entities, triples, context)

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "reproducibility_notes": repro_notes,
            "overall_assessment": overall
        }

    def _fill_templates(self, templates: List[str], context: Dict, n: int = 2) -> List[str]:
        available = [t for t in templates if self._can_fill(t, context)]
        if not available:
            # 使用随机上下文补齐缺失字段
            return [t.format(**{**context, **self._random_context(t)}) for t in templates[:n]]
        selected = random.sample(available, min(n, len(available)))
        return [t.format(**context) for t in selected]

    def _can_fill(self, template: str, context: Dict) -> bool:
        for key in context:
            if "{" + key + "}" in template:
                return True
        return True

    def _random_context(self, template: str) -> Dict:
        return {
            "dataset": random.choice(DEFAULT_DATASETS),
            "other_dataset": random.choice(DEFAULT_DATASETS),
            "metric": random.choice(DEFAULT_METRICS),
            "method": random.choice(DEFAULT_METHODS),
            "model": random.choice(DEFAULT_MODELS),
            "baseline": random.choice(DEFAULT_MODELS),
            "analysis": "ablation",
            "concern": "generalization",
            "missing_item": "error analysis",
            "dataset_2": random.choice(DEFAULT_DATASETS),
            "task": random.choice(DEFAULT_TASKS),
            "value": "competitive results",
            "limitation": "computational cost",
            "hyperparam": "batch size",
        }

    def _generate_overall(self, paper_info: Dict, entities: List[Dict],
                          triples: List[Dict], context: Dict) -> str:
        """生成整体评估"""
        quality = paper_info.get("quality_tier", "Acceptable")
        domains = paper_info.get("domains", [])

        if quality == "Acceptable":
            assessment = (
                f"整体来看，本文在 {context['task']} 领域提出了有价值的 {context['method']} 方法，"
                f"在 {context['dataset']} 上取得了 {context['metric']} 的提升。"
            )
        elif quality == "Borderline":
            assessment = (
                f"论文在 {context['task']} 领域的贡献有一定价值，"
                f"但实验验证方面存在不足，特别是在 {context['limitation']} 方面需要补充论证。"
            )
        else:
            assessment = (
                f"论文在方法创新性和实验完整性方面存在较大不足，"
                f"建议在 {context['analysis']} 和 {context['limitation']} 方面做出重大改进后重新投稿。"
            )

        if len(entities) > 5:
            assessment += f" 论文包含 {len(entities)} 个关键实体和 {len(triples)} 个关系三元组，信息抽取覆盖较好。"
        else:
            assessment += " 建议进一步丰富论文的技术细节描述。"

        return assessment


def generate_review_draft(entities: List[Dict], triples: List[Dict],
                          paper_info: Dict = None) -> Dict:
    """便捷接口"""
    generator = ReviewGenerator()
    return generator.generate(entities, triples, paper_info)
