"""
专用 Agent 实现

每个 Agent 封装一个 NLP 模块的调用逻辑:
    - ClassifierAgent: 模块一 (文本分类)
    - ExtractorAgent: 模块二 (信息抽取)
    - AnalystAgent: 模块三 (摘要 + 实验分析)
    - RetrieverAgent: 模块四 (语义检索)
    - ReviewerAgent: 审稿报告生成 + 反思校验
"""
from typing import Dict, List

from .base import BaseAgent
from .memory import WorkingMemory, ReflectionMemory


class ClassifierAgent(BaseAgent):
    """论文分类 Agent — 调用模块一"""

    def __init__(self, classifier, registry=None):
        super().__init__("ClassifierAgent", "论文多维度分类器",
                         tools=["classify_paper"], registry=registry)
        self.classifier = classifier

    def execute(self, memory: WorkingMemory) -> Dict:
        paper_text = memory.get("paper_text", "")

        self.think("需要理解论文的领域类型、方法类型和质量等级，为后续任务分配提供依据")

        result = self.classifier.classify(paper_text)

        self.observe(
            f"领域={result.get('domains', [])}, "
            f"方法={result.get('method_type', 'Unknown')}, "
            f"质量={result.get('quality_tier', 'Unknown')}"
        )

        memory.set("classification", result)
        memory.set("domains", result.get("domains", []))
        memory.set("quality_tier", result.get("quality_tier", "Acceptable"))

        self.think("分类完成。根据结果，这是一篇实验型论文，可并行调度抽取和检索")
        return result


class ExtractorAgent(BaseAgent):
    """信息抽取 Agent — 调用模块二"""

    def __init__(self, extractor, registry=None):
        super().__init__("ExtractorAgent", "科研信息抽取引擎",
                         tools=["extract_information"], registry=registry)
        self.extractor = extractor

    def execute(self, memory: WorkingMemory) -> Dict:
        paper_text = memory.get("paper_text", "")

        self.think("需要从论文中抽取关键实体(MODEL/DATASET/METRIC/METHOD/TASK)和关系三元组")

        result = self.extractor.extract_information(paper_text)
        entities = result.get("entities", [])
        triples = result.get("triples", [])

        self.observe(
            f"抽取到 {len(entities)} 个实体, {len(triples)} 个三元组"
        )

        # 分类统计
        entity_types = {}
        for e in entities:
            t = e.get("type", "UNKNOWN")
            entity_types[t] = entity_types.get(t, 0) + 1
        if entity_types:
            self.think(f"实体类型分布: {entity_types}")

        memory.set("extraction", result)
        memory.set("entities", entities)
        memory.set("triples", triples)

        # 检查抽取充分性
        if len(entities) < 3:
            self.think("抽取到的实体较少，可能需要更细粒度的全文分析")
        if len(triples) == 0:
            self.think("未抽到关系三元组，审稿时需注意信息完整性")

        return result


class AnalystAgent(BaseAgent):
    """实验分析 Agent — 调用模块三 + 分析逻辑"""

    def __init__(self, summarizer=None, registry=None):
        super().__init__("AnalystAgent", "实验严谨性与对比完整性分析",
                         tools=["generate_summary"], registry=registry)
        self.summarizer = summarizer

    def execute(self, memory: WorkingMemory) -> Dict:
        paper_text = memory.get("paper_text", "")
        entities = memory.get("entities", [])
        triples = memory.get("triples", [])
        classification = memory.get("classification", {})

        self.think("需要分析实验设计是否严谨，对比是否充分")

        # 实验完整性分析
        datasets = [e["text"] for e in entities if e["type"] == "DATASET"]
        metrics_list = [e["text"] for e in entities if e["type"] == "METRIC"]
        methods = [e["text"] for e in entities if e["type"] == "METHOD"]
        models = [e for e in entities if e.get("type") == "MODEL"]

        analysis = {
            "dataset_count": len(datasets),
            "datasets": datasets,
            "metrics": metrics_list,
            "methods": methods,
            "model_count": len(models),
            "concerns": [],
            "strengths": [],
        }

        # 实验完整性检查
        if len(datasets) == 0:
            analysis["concerns"].append("未识别到具体数据集，实验可能缺少标准化评估")
        elif len(datasets) == 1:
            analysis["concerns"].append(f"仅在一个数据集({datasets[0]})上验证，缺乏跨域泛化评估")
        elif len(datasets) >= 3:
            analysis["strengths"].append(f"在{len(datasets)}个数据集上验证，实验覆盖面较好")

        # 对比公正性检查
        if len(models) < 2:
            analysis["concerns"].append("未识别到足够的对比模型，baseline对比可能不充分")

        # 方法检查
        if len(methods) == 0:
            analysis["concerns"].append("方法描述不够清晰，可能影响可复现性")

        self.observe(
            f"数据集={len(datasets)}个, 指标={len(metrics_list)}个, "
            f"关注点={len(analysis['concerns'])}个"
        )

        # 生成摘要 (可选，如果加载了summarizer)
        summary_result = None
        if self.summarizer:
            self.think("调用摘要生成器生成结构化摘要")
            summary_result = self.summarizer.generate_summary(
                paper_text, entities, triples, classification
            )
            analysis["structured_summary"] = summary_result.get("structured_summary", {})
            analysis["extractive_skeleton"] = summary_result.get("extractive_skeleton", [])

        memory.set("analysis", analysis)
        memory.set("summary", summary_result)
        return analysis


class RetrieverAgent(BaseAgent):
    """论文检索 Agent — 调用模块四"""

    def __init__(self, retriever=None, registry=None):
        super().__init__("RetrieverAgent", "语义检索与相关工作推荐",
                         tools=["semantic_search", "detect_similarity"], registry=registry)
        self.retriever = retriever

    def execute(self, memory: WorkingMemory) -> Dict:
        paper_text = memory.get("paper_text", "")
        domains = memory.get("domains", [])

        self.think(f"需要在论文库中检索与当前论文语义相似的工作 (领域: {domains})")

        result = {"similar_papers": [], "potential_overlap": [], "citation_recommendations": []}

        if self.retriever and self.retriever.index.size > 0:
            # 相似论文检索
            self.think("执行语义检索，查找相关工作")
            similar = self.retriever.semantic_search(paper_text, top_k=10)
            result["similar_papers"] = similar

            # 重复性检测
            self.think("检测是否存在高度重复的已发表工作")
            overlap = self.retriever.detect_similarity(paper_text, threshold=0.85)
            result["potential_overlap"] = overlap

            self.observe(
                f"检索到 {len(similar)} 篇相关工作, "
                f"{len(overlap)} 篇高度相似论文"
            )

            if len(overlap) > 0:
                self.think("存在高度相似论文，需在审稿中标记重复性风险")
        else:
            self.think("语义索引未初始化，跳过检索步骤")
            self.observe("检索跳过 (索引为空)")

        memory.set("retrieval", result)
        return result


class ReviewerAgent(BaseAgent):
    """审稿报告生成 Agent — 综合所有信息生成最终报告"""

    def __init__(self, review_generator=None, registry=None):
        super().__init__("ReviewerAgent", "审稿报告生成与反思校验",
                         tools=["generate_summary"], registry=registry)
        self.review_generator = review_generator
        self.reflection = ReflectionMemory()

    def execute(self, memory: WorkingMemory) -> Dict:
        self.think("所有子任务已完成，现在综合所有信息生成审稿报告")

        # 汇总所有信息
        classification = memory.get("classification", {})
        entities = memory.get("entities", [])
        triples = memory.get("triples", [])
        analysis = memory.get("analysis", {})
        retrieval = memory.get("retrieval", {})
        summary = memory.get("summary") or {}

        # 构建审稿报告
        report = {
            "paper_summary": {
                "domains": classification.get("domains", []),
                "method_type": classification.get("method_type", "Unknown"),
                "quality_tier": classification.get("quality_tier", "Acceptable"),
            },
            "structured_summary": summary.get("structured_summary", analysis.get("structured_summary", {})),
            "extraction_stats": {
                "entities": len(entities),
                "triples": len(triples),
                "entity_types": self._count_types(entities),
            },
            "experiment_analysis": {
                "datasets": analysis.get("datasets", []),
                "metrics": analysis.get("metrics", []),
                "concerns": analysis.get("concerns", []),
                "strengths": analysis.get("strengths", []),
            },
            "similarity_check": {
                "similar_papers": len(retrieval.get("similar_papers", [])),
                "potential_overlap": len(retrieval.get("potential_overlap", [])),
            },
            "review_draft": {},
            "reflection_notes": [],
        }

        # 生成审稿意见
        if self.review_generator:
            self.think("基于抽取结果生成审稿意见草稿")
            paper_info = {
                "quality_tier": classification.get("quality_tier", "Acceptable"),
                "domains": classification.get("domains", []),
            }
            review = self.review_generator.generate(entities, triples, paper_info)
            report["review_draft"] = review
            report["overall_assessment"] = review.get("overall_assessment", "")
        else:
            report["review_draft"] = self._generate_basic_review(analysis, entities, triples)

        # 反思校验
        self.think("执行审稿完整性自检...")
        checklist = self.reflection.get_checklist()
        for item in checklist:
            self.reflect(item)

        # 补充反思备注
        reflection_notes = []
        if len(analysis.get("concerns", [])) > 3:
            reflection_notes.append("实验存在多处不足，建议给出具体改进建议而非笼统拒稿")
        if len(entities) < 3:
            reflection_notes.append("信息抽取不够充分，可能是论文本身技术细节不足")
        if retrieval.get("potential_overlap"):
            reflection_notes.append("存在高度相似论文，建议作者说明与已有工作的差异")
        report["reflection_notes"] = reflection_notes

        self.observe(f"报告生成完成，包含 {len(reflection_notes)} 条反思备注")
        memory.set("final_report", report)
        return report

    def _count_types(self, entities: List[Dict]) -> Dict:
        counts = {}
        for e in entities:
            t = e.get("type", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _generate_basic_review(self, analysis: Dict, entities: List[Dict],
                               triples: List[Dict]) -> Dict:
        """无 ReviewGenerator 时的基础审稿"""
        concerns = analysis.get("concerns", [])
        strengths = analysis.get("strengths", [])
        return {
            "strengths": strengths or ["论文选题有一定价值"],
            "weaknesses": concerns or ["实验验证有待加强"],
            "suggestions": ["建议补充更多对比实验"],
            "overall_assessment": "论文整体质量中等，建议修改后接收。"
        }
