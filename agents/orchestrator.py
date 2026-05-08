"""
系统编排器 — 组装所有模块和Agent，提供统一入口

职责:
    1. 初始化所有 NLP 模块 (分类/抽取/摘要/检索)
    2. 创建 Agent 实例并注册到 Coordinator
    3. 提供 run_pipeline() 端到端审稿接口
    4. 输出结构化审稿报告

架构:
    NLP模块 → Agent封装 → Coordinator调度 → 结构化报告
"""
import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .tools import ToolRegistry, get_registry, register_tool
from .memory import WorkingMemory, KnowledgeGraph, ReflectionMemory
from .base import BaseAgent
from .specialized import ClassifierAgent, ExtractorAgent, AnalystAgent, RetrieverAgent, ReviewerAgent
from .coordinator import CoordinatorAgent


@dataclass
class PipelineConfig:
    """Pipeline 配置"""
    # 模块一: 分类器
    classifier_model_path: str = "checkpoints/multitask/best_model.pt"
    classifier_model_type: str = "multitask"

    # 模块二: 信息抽取
    ner_model_path: str = "checkpoints/ner/best_model.pt"
    re_model_path: str = "checkpoints/re/best_model.pt"

    # 模块三: 摘要生成
    generative_model_path: Optional[str] = None
    generative_model_name: str = "facebook/bart-base"

    # 模块四: 语义检索
    retrieval_index_dir: Optional[str] = None
    retrieval_encoder: str = "allenai/specter"

    # 知识图谱
    knowledge_graph_path: str = "knowledge_graph.json"

    # 设备
    device: str = "cuda"


class PaperReviewOrchestrator:
    """论文审稿系统编排器"""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.registry = get_registry()
        self.knowledge_graph = KnowledgeGraph(self.config.knowledge_graph_path)
        self.classifier = None
        self.extractor = None
        self.summarizer = None
        self.retriever = None
        self.coordinator: Optional[CoordinatorAgent] = None
        self._initialized = False

    def initialize(self):
        """初始化所有模块和Agent"""
        if self._initialized:
            return
        print("=" * 60)
        print("初始化论文审稿多Agent系统")
        print("=" * 60)

        # 加载知识图谱
        self.knowledge_graph.load()

        # 模块一: 分类器
        print("\n[1/4] 加载分类器 (模块一)...")
        try:
            from models.classifier.scibert_classifier import PaperClassifier
            self.classifier = PaperClassifier(
                model_path=self.config.classifier_model_path,
                model_type=self.config.classifier_model_type,
                device=self.config.device
            )
            self._register_classify_tool()
            print("  分类器加载成功")
        except Exception as e:
            print(f"  分类器加载失败: {e}")

        # 模块二: 信息抽取
        print("\n[2/4] 加载信息抽取器 (模块二)...")
        try:
            if os.path.exists(self.config.ner_model_path) and os.path.exists(self.config.re_model_path):
                from models.extraction.paper_extractor import PaperExtractor
                self.extractor = PaperExtractor(
                    ner_model_path=self.config.ner_model_path,
                    re_model_path=self.config.re_model_path,
                    device=self.config.device
                )
                self._register_extract_tool()
                print("  信息抽取器加载成功")
            else:
                print("  抽取模型未找到，使用占位模式")
        except Exception as e:
            print(f"  信息抽取器加载失败: {e}")

        # 模块三: 摘要生成
        print("\n[3/4] 加载摘要生成器 (模块三)...")
        try:
            from models.summarizer.paper_summarizer import PaperSummarizer
            self.summarizer = PaperSummarizer(
                generative_model_path=self.config.generative_model_path,
                generative_model_name=self.config.generative_model_name,
                device=self.config.device
            )
            print("  摘要生成器加载成功")
        except Exception as e:
            print(f"  摘要生成器加载失败: {e}")

        # 模块四: 语义检索
        print("\n[4/4] 加载检索引擎 (模块四)...")
        try:
            if self.config.retrieval_index_dir and os.path.exists(
                os.path.join(self.config.retrieval_index_dir, "papers.index")
            ):
                from models.retrieval import PaperRetriever
                self.retriever = PaperRetriever(
                    encoder_model=self.config.retrieval_encoder,
                    index_dir=self.config.retrieval_index_dir,
                    device=self.config.device
                )
                self._register_retrieval_tools()
                print(f"  检索引擎加载成功 ({self.retriever.index.size} 篇论文)")
            else:
                print("  检索索引未构建，跳过")
        except Exception as e:
            print(f"  检索引擎加载失败: {e}")

        # 创建 Agent
        print("\n创建 Agent 团队...")
        coordinator = CoordinatorAgent(knowledge_graph=self.knowledge_graph)

        coordinator.register_agent("ClassifierAgent", ClassifierAgent(self.classifier))
        coordinator.register_agent("ExtractorAgent", ExtractorAgent(self.extractor))
        coordinator.register_agent("AnalystAgent", AnalystAgent(self.summarizer))
        coordinator.register_agent("RetrieverAgent", RetrieverAgent(self.retriever))
        coordinator.register_agent("ReviewerAgent", ReviewerAgent(
            self.summarizer.review_generator if self.summarizer else None
        ))

        self.coordinator = coordinator
        self._initialized = True

        print(f"\n系统初始化完成。Coordinator 管理 {len(coordinator.agents)} 个 Agent")

    def run_review(self, paper_text: str, paper_id: str = "",
                   paper_title: str = "") -> Dict:
        """端到端审稿入口"""
        if not self._initialized:
            self.initialize()

        print("\n" + "=" * 60)
        print(f"开始审稿: {paper_title or paper_id or 'Untitled'}")
        print("=" * 60)

        memory = WorkingMemory(paper_id=paper_id or f"paper_{int(time.time())}")
        memory.set("paper_text", paper_text)
        memory.set("paper_title", paper_title)

        start_time = time.time()
        final_report = self.coordinator.run(memory)
        elapsed = time.time() - start_time

        # 添加元信息
        final_report["meta"] = {
            "paper_id": memory.paper_id,
            "paper_title": paper_title,
            "review_time": round(elapsed, 2),
            "agent_trace": [{
                "thought": step.thought,
                "action": step.action,
                "observation": step.observation,
                "elapsed": step.elapsed
            } for step in self.coordinator.trace]
        }

        print(f"\n审稿完成! 总用时: {elapsed:.1f}s")
        return final_report

    def _register_classify_tool(self):
        if not self.classifier:
            return
        register_tool(
            name="classify_paper",
            description="对论文进行领域、方法类型、质量等级分类",
            parameters={
                "paper_text": {"type": "string", "description": "论文标题+摘要文本"}
            },
            func=lambda paper_text: self.classifier.classify(paper_text),
            module="NLP-Module-1"
        )

    def _register_extract_tool(self):
        if not self.extractor:
            return
        register_tool(
            name="extract_information",
            description="从论文文本中抽取科研实体和关系三元组",
            parameters={
                "paper_text": {"type": "string", "description": "论文全文文本"}
            },
            func=lambda paper_text: self.extractor.extract_information(paper_text),
            module="NLP-Module-2"
        )

    def _register_retrieval_tools(self):
        if not self.retriever:
            return
        register_tool(
            name="semantic_search",
            description="检索语义相似论文 (相关工作推荐)",
            parameters={
                "query_text": {"type": "string", "description": "查询论文文本"},
                "top_k": {"type": "integer", "description": "返回数量", "default": 5}
            },
            func=lambda query_text, top_k=5: self.retriever.semantic_search(query_text, top_k),
            module="NLP-Module-4"
        )
        register_tool(
            name="detect_similarity",
            description="检测高相似度论文 (重复性检查)",
            parameters={
                "query_text": {"type": "string", "description": "待检测论文文本"},
                "threshold": {"type": "float", "description": "相似度阈值", "default": 0.85}
            },
            func=lambda query_text, threshold=0.85: self.retriever.detect_similarity(query_text, threshold),
            module="NLP-Module-4"
        )

    def save_report(self, report: Dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"审稿报告已保存: {output_path}")

    def save_knowledge_graph(self):
        self.knowledge_graph.save()
        print(f"知识图谱已保存: {self.config.knowledge_graph_path}")
