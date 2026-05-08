"""
工具注册中心与 Function Calling 调度

设计:
    - 全局工具注册表: 所有 NLP 模块的 Agent 接口注册为 Tool
    - Function Calling Schema: JSON Schema 格式的工具定义
    - 工具执行调度: 根据 Agent 请求调用对应工具
"""
import json
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict  # JSON Schema for parameters
    func: Callable = None
    module: str = ""

    def to_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """全局工具注册中心"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict]:
        return [t.to_schema() for t in self._tools.values()]

    def execute(self, tool_name: str, params: Dict) -> Dict:
        """执行工具调用"""
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            start = time.time()
            result = tool.func(**params)
            elapsed = time.time() - start
            return {"tool": tool_name, "result": result, "elapsed": round(elapsed, 3)}
        except Exception as e:
            return {"tool": tool_name, "error": str(e)}


# 全局单例
_registry = ToolRegistry()


def register_tool(name: str, description: str, parameters: Dict, func: Callable = None,
                  module: str = ""):
    """装饰器/直接注册工具"""
    tool = Tool(name=name, description=description, parameters=parameters,
                func=func, module=module)
    _registry.register(tool)
    return tool


def get_registry() -> ToolRegistry:
    return _registry


def execute_tool(tool_name: str, **params) -> Dict:
    """便捷执行"""
    return _registry.execute(tool_name, params)
