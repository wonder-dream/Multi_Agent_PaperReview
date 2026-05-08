"""
Agent 基类 - ReAct 推理循环

所有 Agent 的基类，提供:
    - ReAct 推理跟踪 (Thought → Action → Observation)
    - 工具调用接口
    - 结果处理和日志
"""
import time
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .tools import ToolRegistry, get_registry
from .memory import WorkingMemory


@dataclass
class ReActStep:
    """ReAct 推理步骤记录"""
    thought: str
    action: str = ""
    observation: str = ""
    result: Any = None
    elapsed: float = 0.0


class BaseAgent(ABC):
    """Agent 基类"""

    def __init__(self, name: str, role: str, tools: List[str] = None,
                 registry: ToolRegistry = None):
        self.name = name
        self.role = role
        self.tool_names = tools or []
        self.registry = registry or get_registry()
        self.memory: Optional[WorkingMemory] = None
        self.trace: List[ReActStep] = []

    def think(self, thought: str):
        """记录思考"""
        step = ReActStep(thought=thought)
        self.trace.append(step)
        print(f"  [{self.name}] [思考] {thought}")

    def act(self, tool_name: str, **params) -> Dict:
        """执行工具调用"""
        if self.trace:
            self.trace[-1].action = f"{tool_name}({json.dumps(params, ensure_ascii=False)})"

        print(f"  [{self.name}] [行动] 调用 {tool_name}({self._brief_params(params)})")
        t0 = time.time()

        result = self.registry.execute(tool_name, params)

        elapsed = time.time() - t0
        if self.trace:
            self.trace[-1].elapsed = elapsed
            self.trace[-1].result = result

        return result

    def observe(self, observation: str):
        """记录观察"""
        if self.trace:
            self.trace[-1].observation = observation
        print(f"  [{self.name}] [观察] {observation}")

    def reflect(self, question: str) -> str:
        """反思检查"""
        print(f"  [{self.name}] [反思] {question}")
        return question

    @abstractmethod
    def execute(self, memory: WorkingMemory) -> Dict:
        """执行Agent任务，由子类实现"""
        pass

    def run(self, memory: WorkingMemory) -> Dict:
        """运行Agent"""
        print(f"\n{'─'*40}")
        print(f"[{self.name}] ({self.role}) 开始执行")
        print(f"{'─'*40}")

        self.memory = memory
        self.trace = []

        result = self.execute(memory)

        print(f"[{self.name}] 执行完毕")
        return result

    def _brief_params(self, params: Dict) -> str:
        """简短显示参数"""
        items = []
        for k, v in params.items():
            v_str = str(v)
            if len(v_str) > 60:
                v_str = v_str[:57] + "..."
            items.append(f"{k}={v_str}")
        return ", ".join(items)
