"""LangGraph 路由测试，不调用真实模型或网络。"""

import unittest
from types import SimpleNamespace

from agent_graph import build_agent_graph


class FakeToolCall:
    def model_dump(self, exclude_none=True) -> dict:
        return {
            "id": "call-1",
            "type": "function",
            "function": {"name": "list_vehicles", "arguments": "{}"},
        }


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            message = SimpleNamespace(
                content=None,
                tool_calls=[FakeToolCall()],
            )
        else:
            message = SimpleNamespace(
                content="找到 VIN789。",
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


class FakeMcpSession:
    async def call_tool(self, name, arguments):
        return SimpleNamespace(
            isError=False,
            structuredContent={"vins": ["VIN123", "VIN789"]},
            content=[],
        )


class AgentGraphTest(unittest.IsolatedAsyncioTestCase):
    async def test_model_tool_model_end(self):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        graph = build_agent_graph(
            client,
            "fake-model",
            FakeMcpSession(),
            [],
        )
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "找异常车辆"}],
                "tool_rounds": 0,
                "phase": "start",
                "tool_trace": [],
                "final_answer": "",
            }
        )
        self.assertEqual(result["final_answer"], "找到 VIN789。")
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["tool_trace"], ["list_vehicles"])
        self.assertEqual(result["phase"], "model")


if __name__ == "__main__":
    unittest.main()
