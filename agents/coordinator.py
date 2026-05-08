"""
主控 Coordinator Agent — ReAct 任务调度中心

核心职责:
    1. 接收论文，执行任务分解
    2. 按依赖关系调度子 Agent (串行/并行)
    3. 汇总各 Agent 结果
    4. 触发反思循环，补充遗漏分析

ReAct 调度流程:
    输入论文 → [思考] 分类 → [行动] ClassifierAgent
    → [观察] 领域+质量 → [思考] 并行抽取+检索
    → [行动] ExtractorAgent ∥ RetrieverAgent
    → [观察] 实体+相似论文 → [思考] 深度分析
    → [行动] AnalystAgent → [观察] 实验分析
    → [思考] 生成报告 → [行动] ReviewerAgent
    → [反思] 完整性检查 → 输出审稿报告
"""
import json
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseAgent
from .memory import WorkingMemory, KnowledgeGraph


class CoordinatorAgent(BaseAgent):
    """主控 Agent — 论文审稿任务总调度"""

    def __init__(self, agents: Dict[str, BaseAgent] = None, knowledge_graph: KnowledgeGraph = None):
        super().__init__("Coordinator", "多Agent任务编排与审稿调度",
                         tools=["classify_paper", "extract_information", "generate_summary",
                                "semantic_search", "detect_similarity"])
        self.agents = agents or {}
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()

    def register_agent(self, name: str, agent):
        if agent is None:
            self.agents[name] = None
            print(f"[Coordinator] 注册 Agent: {name} (跳过 - 未加载)")
            return
        self.agents[name] = agent
        print(f"[Coordinator] 注册 Agent: {name} ({agent.role})")

    def execute(self, memory: WorkingMemory) -> Dict:
        paper_text = memory.get("paper_text", "")
        paper_id = memory.get("paper_id", "unknown")

        self.think(f"收到论文 [{paper_id}]，开始任务分解。首先需要进行分类以确定论文领域和类型。")

        # Phase 1: 分类 (串行先决条件)
        self.think("Phase 1: 论文分类 — 这是后续所有任务的先决条件")
        self._run_agent("ClassifierAgent", memory)

        classification = memory.get("classification", {})
        domains = classification.get("domains", [])
        quality = classification.get("quality_tier", "Acceptable")

        self.observe(f"论文领域={domains}, 质量等级={quality}")

        # Phase 2: 抽取 + 检索 (可并行)
        self.think("Phase 2: 信息抽取和语义检索互不依赖，可并行执行以提升效率")
        parallel_agents = ["ExtractorAgent", "RetrieverAgent"]
        self._run_parallel(parallel_agents, memory)

        extraction = memory.get("extraction", {})
        retrieval = memory.get("retrieval", {})

        n_entities = len(extraction.get("entities", []))
        n_triples = len(extraction.get("triples", []))
        n_similar = len(retrieval.get("similar_papers", []))

        self.observe(f"抽取: {n_entities}实体/{n_triples}关系, 检索: {n_similar}篇相关工作")

        # Phase 3: 实验分析
        self.think("Phase 3: 基于抽取结果进行实验严谨性深度分析")
        if n_entities > 0:
            self.think(f"抽取结果显示主要使用了 {extraction.get('entities', [{}])[0].get('text', 'N/A')} 等方法，"
                       f"需要检查实验完整性")
        self._run_agent("AnalystAgent", memory)

        analysis = memory.get("analysis", {})
        concerns = analysis.get("concerns", [])
        self.observe(f"实验分析: {len(concerns)}个关注点, {len(analysis.get('strengths', []))}个优势")

        # Phase 4: 生成审稿报告
        self.think("Phase 4: 所有分析完成，调度 ReviewerAgent 生成最终审稿报告")
        self._run_agent("ReviewerAgent", memory)

        # Phase 5: 反思与知识沉淀
        self.think("Phase 5: 反思总结 — 检查是否有遗漏的分析维度")

        final_report = memory.get("final_report", {})
        self._reflect_and_complete(memory, final_report)

        # 知识图谱更新
        entities = memory.get("entities", [])
        triples = memory.get("triples", [])
        paper_info = {
            "paper_id": paper_id,
            "text": paper_text[:500],
            "domains": domains,
            "quality_tier": quality
        }
        self.knowledge_graph.add_paper(paper_info)
        self.knowledge_graph.add_entities(entities)
        self.knowledge_graph.add_triples(triples)

        self.observe(f"审稿完成。知识图谱已更新 (共{self.knowledge_graph.stats()['papers']}篇论文)")

        return final_report

    def _run_agent(self, agent_name: str, memory: WorkingMemory):
        agent = self.agents.get(agent_name)
        if agent:
            agent.run(memory)
        else:
            print(f"  [Coordinator] Agent '{agent_name}' 未注册，跳过")

    def _run_parallel(self, agent_names: List[str], memory: WorkingMemory):
        """并行执行多个 Agent"""
        available = [a for a in agent_names if a in self.agents and self.agents[a] is not None]
        if len(available) == 0:
            return
        if len(available) == 1:
            self._run_agent(available[0], memory)
            return

        print(f"  [Coordinator] 并行调度: {available}")
        # 每个 Agent 使用独立的 memory snapshot 再合并
        with ThreadPoolExecutor(max_workers=len(available)) as executor:
            futures = {}
            for name in available:
                # 创建独立memory副本避免竞争
                agent_memory = WorkingMemory(paper_id=memory.paper_id)
                agent_memory.update(memory.snapshot())
                futures[executor.submit(self.agents[name].run, agent_memory)] = (name, agent_memory)

            for future in as_completed(futures):
                name, agent_mem = futures[future]
                try:
                    future.result()
                    # 合并结果回主memory
                    for key in agent_mem._store:
                        if key not in ("paper_text", "paper_id"):
                            memory.set(key, agent_mem._store[key])
                except Exception as e:
                    print(f"  [Coordinator] {name} 执行异常: {e}")

    def _reflect_and_complete(self, memory: WorkingMemory, report: Dict):
        """反思和完善"""
        notes = report.get("reflection_notes", [])

        # 检查遗漏
        analysis = memory.get("analysis", {})
        retrieval = memory.get("retrieval", {})

        # 补充引用推荐
        if retrieval.get("similar_papers"):
            self.think("引用推荐: 基于检索结果，建议作者关注以下相关工作")
        else:
            self.reflect("是否遗漏了引用完整性分析?")
            notes.append("建议进行相关工作检索以获得更全面的引用建议")

        # 检查伦理声明
        self.reflect("是否遗漏了伦理声明分析?")
        paper_text = memory.get("paper_text", "").lower()
        if any(kw in paper_text for kw in ["human", "privacy", "bias", "ethic", "sensitive"]):
            self.observe("论文涉及潜在伦理问题，需在审稿报告中建议作者补充伦理声明")
            notes.append("论文涉及伦理相关问题，建议补充伦理声明")
        else:
            self.observe("未发现明显伦理问题")

        report["reflection_notes"] = notes
