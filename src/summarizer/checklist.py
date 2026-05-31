"""Rule-based completeness checklist generation from extracted entities."""
from typing import List, Dict


ETHICS_KEYWORDS = ["ethic", "privacy", "bias", "fairness", "human subjects",
                   "irb", "institutional review", "consent", "regulation"]
CODE_KEYWORDS = ["github", "code", "repository", "open.source", "available at",
                 "http", "www.", ".com", ".org", ".io", "reproduc"]
ABLATION_KEYWORDS = ["ablation", "ablate", "remove", "contribution of",
                     "effect of each", "component analysis", "module analysis"]


class ChecklistEngine:
    """Generates completeness checklist from entities and paper text."""

    def generate(self, entities: List[Dict], paper_text: str = "") -> List[Dict]:
        """Generate checklist items based on extracted entities and paper text."""
        text_lower = paper_text.lower()
        items = []

        items.append(self._check_datasets(entities))
        items.append(self._check_baselines(entities))
        items.append(self._check_ablation(text_lower))
        items.append(self._check_code(text_lower))
        items.append(self._check_ethics(text_lower))
        items.append(self._check_metrics(entities))

        return [i for i in items if i is not None]

    def _check_datasets(self, entities: List[Dict]) -> Dict:
        datasets = [e["text"] for e in entities if e["type"] == "DATASET"]
        n = len(datasets)
        if n >= 3:
            status = "ok"
            detail = f"在 {n} 个数据集上验证 ({', '.join(datasets)})"
        elif n >= 1:
            status = "partial"
            detail = f"仅在 {n} 个数据集上验证 ({', '.join(datasets)})，建议补充更多数据集"
        else:
            status = "missing"
            detail = "未检测到明确的数据集信息"
        return {"category": "数据集多样性", "status": status, "detail": detail}

    def _check_baselines(self, entities: List[Dict]) -> Dict:
        models = [e["text"] for e in entities if e["type"] == "MODEL"]
        n = len(models)
        if n >= 3:
            status = "ok"
            detail = f"对比了 {n} 个模型 ({', '.join(models)})"
        elif n >= 1:
            status = "partial"
            detail = f"仅涉及 {n} 个模型 ({', '.join(models)})，建议增加 SOTA 对比"
        else:
            status = "missing"
            detail = "未检测到明确的模型命名"
        return {"category": "Baseline对比", "status": status, "detail": detail}

    def _check_ablation(self, text_lower: str) -> Dict:
        if any(kw in text_lower for kw in ABLATION_KEYWORDS):
            return {"category": "消融实验", "status": "ok", "detail": "检测到消融实验相关描述"}
        return {"category": "消融实验", "status": "missing", "detail": "未发现消融实验或各模块贡献分析"}

    def _check_code(self, text_lower: str) -> Dict:
        if any(kw in text_lower for kw in CODE_KEYWORDS):
            return {"category": "代码开源", "status": "ok", "detail": "检测到代码或数据链接"}
        return {"category": "代码开源", "status": "missing", "detail": "未找到代码链接或开源声明"}

    def _check_ethics(self, text_lower: str) -> Dict:
        if any(kw in text_lower for kw in ETHICS_KEYWORDS):
            return {"category": "伦理声明", "status": "ok", "detail": "包含伦理相关讨论"}
        return {"category": "伦理声明", "status": "unchecked", "detail": "未发现伦理声明或隐私讨论"}

    def _check_metrics(self, entities: List[Dict]) -> Dict:
        metrics = [e["text"] for e in entities if e["type"] == "METRIC"]
        n = len(metrics)
        if n >= 3:
            return {"category": "指标完整性", "status": "ok", "detail": f"使用了 {n} 种指标 ({', '.join(metrics)})"}
        elif n >= 1:
            return {"category": "指标完整性", "status": "partial",
                    "detail": f"仅使用了 {n} 种指标 ({', '.join(metrics)})，建议增加评估维度"}
        return {"category": "指标完整性", "status": "unchecked",
                "detail": "未检测到明确的评估指标"}
