"""模型 Provider 的最小公共接口。"""

from typing import Protocol


class LLMProvider(Protocol):
    name: str

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        """返回 OpenAI 消息形状：role、content、可选 tool_calls。"""
        ...
