"""LangGraph 路由测试，不调用真实模型或网络。"""

import json
import unittest
from types import SimpleNamespace

from agent_graph import build_agent_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class FakeToolCall:
    def __init__(self, name="list_vehicles", arguments=None):
        self.name = name
        self.arguments = arguments or {}

    def model_dump(self, exclude_none=True) -> dict:
        return {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


class FakeProvider:
    name = "fake"

    def __init__(self, tool_name="list_vehicles", arguments=None):
        self.calls = 0
        self.seen_messages = []
        self.tool_name = tool_name
        self.arguments = arguments or {}

    def generate(self, messages, tools):
        self.calls += 1
        self.seen_messages.append(messages)
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    FakeToolCall(
                        self.tool_name,
                        self.arguments,
                    ).model_dump()
                ],
            }
        return {"role": "assistant", "content": "找到 VIN789。"}


class FakeMcpSession:
    async def call_tool(self, name, arguments):
        return SimpleNamespace(
            isError=False,
            structuredContent={"vins": ["VIN123", "VIN789"]},
            content=[],
        )


class AgentGraphTest(unittest.IsolatedAsyncioTestCase):
    async def test_model_tool_model_end(self):
        graph = build_agent_graph(
            FakeProvider(),
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

    async def test_checkpoint_restores_current_vin_next_turn(self):
        provider = FakeProvider("analyze_vin", {"vin": "VIN789"})
        config = {"configurable": {"thread_id": "memory-test"}}
        async with AsyncSqliteSaver.from_conn_string(
            ":memory:"
        ) as checkpointer:
            graph = build_agent_graph(
                provider,
                FakeMcpSession(),
                [],
                checkpointer=checkpointer,
            )
            initial = {
                "messages": [{"role": "user", "content": "分析VIN789"}],
                "tool_rounds": 0,
                "phase": "start",
                "tool_trace": [],
                "final_answer": "",
            }
            first = await graph.ainvoke(initial, config)
            self.assertEqual(first["current_vin"], "VIN789")

            second = await graph.ainvoke(
                {
                    "messages": [{"role": "user", "content": "它呢？"}],
                    "tool_rounds": 0,
                    "phase": "start",
                    "tool_trace": [],
                    "final_answer": "",
                },
                config,
            )
            self.assertEqual(second["current_vin"], "VIN789")
            self.assertIn(
                "当前会话关注的车辆是 VIN789",
                provider.seen_messages[-1][0]["content"],
            )


if __name__ == "__main__":
    unittest.main()
