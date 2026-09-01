"""LangGraph 路由测试，不调用真实模型或网络。"""

import json
import unittest
from types import SimpleNamespace

from agent_graph import build_agent_graph, valid_model_messages


class FakeToolCall:
    def __init__(self, name="search_style_corpus", arguments=None):
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

    def __init__(self, tool_name="search_style_corpus", arguments=None):
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
        return {"role": "assistant", "content": "这是润色后的段落。"}


class FakeMcpSession:
    async def call_tool(self, name, arguments):
        return SimpleNamespace(
            isError=False,
            structuredContent={"results": [{"source": "paper.pdf", "location": "p.2"}]},
            content=[],
        )


class SummaryProvider:
    name = "summary-fake"

    def __init__(self):
        self.calls = 0
        self.seen_messages = []

    def generate(self, messages, tools):
        self.calls += 1
        self.seen_messages.append(messages)
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "已确认：论文讨论纵向研究设计。",
            }
        return {"role": "assistant", "content": "继续润色方法部分。"}


class AgentGraphTest(unittest.IsolatedAsyncioTestCase):
    def test_interrupted_tool_call_is_removed_from_restored_history(self):
        messages = [
            {"role": "user", "content": "列出论文"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "dangling", "function": {"name": "list_corpus_sources",
                "arguments": "{}"}}]},
            {"role": "user", "content": "再试一次"},
        ]
        self.assertEqual(valid_model_messages(messages), [messages[0], messages[2]])

    def test_completed_tool_call_is_preserved(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "complete", "function": {"name": "list_corpus_sources",
                "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "complete", "content": "{}"},
        ]
        self.assertEqual(valid_model_messages(messages), messages)

    async def test_summary_compacts_model_context(self):
        provider = SummaryProvider()
        graph = build_agent_graph(
            provider,
            FakeMcpSession(),
            [],
            summary_trigger=3,
            recent_message_count=2,
        )
        result = await graph.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": f"消息{i}"}
                    for i in range(5)
                ],
                "tool_rounds": 0,
                "phase": "start",
                "tool_trace": [],
                "final_answer": "",
            }
        )
        self.assertEqual(result["summary_cursor"], 3)
        self.assertEqual(result["summary_count"], 1)
        self.assertIn("纵向研究设计", result["conversation_summary"])
        model_messages = provider.seen_messages[1]
        self.assertEqual(len(model_messages), 3)
        self.assertIn("历史会话摘要", model_messages[0]["content"])
        self.assertEqual(model_messages[1]["content"], "消息3")
        self.assertEqual(model_messages[2]["content"], "消息4")

    async def test_model_tool_model_end(self):
        graph = build_agent_graph(
            FakeProvider(),
            FakeMcpSession(),
            [],
        )
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "润色这段引言"}],
                "tool_rounds": 0,
                "phase": "start",
                "tool_trace": [],
                "final_answer": "",
            }
        )
        self.assertEqual(result["final_answer"], "这是润色后的段落。")
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["tool_trace"], ["search_style_corpus"])
        self.assertEqual(result["phase"], "model")

if __name__ == "__main__":
    unittest.main()
