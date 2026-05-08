"""
Agent 记忆系统

三种记忆类型:
    1. 短期记忆 (WorkingMemory): 当前论文处理上下文、各 Agent 中间结果
    2. 长期记忆 (KnowledgeGraph): 论文知识图谱, Neo4j/JSON 存储
    3. 反思记忆 (ReflectionMemory): 审稿模式, 常见错误与修正策略
"""
import json
from typing import Dict, List, Any


class WorkingMemory:
    """短期工作记忆 - 当前论文处理状态"""

    def __init__(self, paper_id: str = ""):
        self.paper_id = paper_id
        self._store: Dict[str, Any] = {}
        self._history: List[Dict] = []

    def set(self, key: str, value: Any):
        self._store[key] = value
        self._history.append({"action": "set", "key": key})

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def update(self, data: Dict):
        self._store.update(data)

    def has(self, key: str) -> bool:
        return key in self._store

    def snapshot(self) -> Dict:
        return dict(self._store)

    def summary(self) -> str:
        keys = list(self._store.keys())
        return f"WorkingMemory({len(keys)} keys: {keys})"

    def clear(self):
        self._store.clear()
        self._history.clear()


class KnowledgeGraph:
    """长期记忆 - 科研知识图谱"""

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path
        self.entities: List[Dict] = []
        self.relations: List[Dict] = []
        self.papers: List[Dict] = []
        self._review_patterns: Dict[str, List[str]] = {}

    def add_paper(self, paper_data: Dict):
        self.papers.append(paper_data)

    def add_entities(self, entities: List[Dict]):
        self.entities.extend(entities)

    def add_triples(self, triples: List[Dict]):
        self.relations.extend(triples)

    def add_review_pattern(self, category: str, observation: str):
        if category not in self._review_patterns:
            self._review_patterns[category] = []
        self._review_patterns[category].append(observation)

    def get_patterns(self, category: str) -> List[str]:
        return self._review_patterns.get(category, [])

    def stats(self) -> Dict:
        return {
            "papers": len(self.papers),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "patterns": {k: len(v) for k, v in self._review_patterns.items()}
        }

    def save(self, path: str = None):
        save_path = path or self.storage_path
        if not save_path:
            return
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "entities": self.entities,
                "relations": self.relations,
                "papers": self.papers,
                "review_patterns": self._review_patterns
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: str = None):
        load_path = path or self.storage_path
        if not load_path:
            return
        import os
        if not os.path.exists(load_path):
            return
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entities = data.get("entities", [])
        self.relations = data.get("relations", [])
        self.papers = data.get("papers", [])
        self._review_patterns = data.get("review_patterns", {})


class ReflectionMemory:
    """反思记忆 - 审稿经验积累"""

    COMMON_PATTERNS = {
        "missing_dataset": [
            "实验仅在单一数据集上进行，缺乏跨域验证",
            "未在标准 benchmark 上进行充分对比",
        ],
        "missing_baseline": [
            "缺少与 SOTA 方法的公平对比",
            "baseline 选择过于陈旧或过于简单",
        ],
        "missing_ablation": [
            "未进行消融实验分析各模块贡献",
            "超参数选择缺乏敏感性分析",
        ],
        "overclaim": [
            "结论超出实验支持范围",
            "未讨论方法局限性和适用边界",
        ],
        "reproducibility": [
            "未公开代码或数据",
            "实验细节描述不充分，难以复现",
        ],
    }

    def __init__(self):
        self.checklist = []
        self.observations: Dict[str, List[str]] = {}

    def check(self, category: str, context: Dict) -> List[str]:
        """根据上下文检查常见问题"""
        issues = []
        patterns = self.COMMON_PATTERNS.get(category, [])
        for p in patterns:
            issues.append(p)
        return issues

    def learn(self, category: str, observation: str):
        """从本次审稿中学习"""
        if category not in self.observations:
            self.observations[category] = []
        self.observations[category].append(observation)

    def get_checklist(self) -> List[str]:
        """获取审稿完整性检查清单"""
        return [
            "论文领域与方法是否已正确分类?",
            "关键实体和关系是否已充分抽取?",
            "实验完整性: 数据集数量/多样性是否充分?",
            "对比公正性: baseline 选择是否合理?",
            "可复现性: 代码/数据是否公开?",
            "局限性: 作者是否诚实地讨论了局限?",
            "伦理声明: 是否涉及伦理问题需要披露?",
        ]
