"""LangGraph 状态图：DeepSeek 决策，MCP 执行工具。"""

import json
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

MAX_TOOL_ROUNDS = 5


class AgentState(TypedDict):
    """图中每个节点共享和更新的状态。"""

    messages: Annotated[list[dict], operator.add]
    tool_rounds: int
    phase: str
    tool_trace: Annotated[list[str], operator.add]
    final_answer: str


def _assistant_message(message) -> dict:
    result = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [
            tool_call.model_dump(exclude_none=True)
            for tool_call in message.tool_calls
        ]
    return result


def build_agent_graph(client, model: str, mcp_session, tools):
    """注入模型与 MCP 会话，然后编译可执行状态图。"""

    def model_node(state: AgentState) -> dict:
        print(f"[graph:model] 第 {state['tool_rounds'] + 1} 次决策")
        response = client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        return {
            "messages": [_assistant_message(message)],
            "phase": "model",
            "final_answer": message.content or "",
        }

    async def mcp_tools_node(state: AgentState) -> dict:
        tool_messages = []
        trace = []
        for tool_call in state["messages"][-1].get("tool_calls", []):
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = await mcp_session.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(f"MCP Tool {name} 调用失败")

            if result.structuredContent is not None:
                output = json.dumps(
                    result.structuredContent,
                    ensure_ascii=False,
                )
            else:
                output = "\n".join(
                    item.text
                    for item in result.content
                    if getattr(item, "type", None) == "text"
                )
            print(f"[graph:mcp_tools] {name}({arguments}) -> {output}")
            trace.append(name)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": output,
                }
            )
        return {
            "messages": tool_messages,
            "tool_rounds": state["tool_rounds"] + 1,
            "phase": "mcp_tools",
            "tool_trace": trace,
        }

    def limit_node(state: AgentState) -> dict:
        answer = (
            f"为防止无限循环，Agent 已在 {state['tool_rounds']} "
            "轮工具调用后停止。"
        )
        print("[graph:limit] 达到工具轮数上限")
        return {
            "messages": [{"role": "assistant", "content": answer}],
            "phase": "limit",
            "final_answer": answer,
        }

    def route_after_model(
        state: AgentState,
    ) -> Literal["tools", "limit", "end"]:
        has_tool_calls = bool(state["messages"][-1].get("tool_calls"))
        if not has_tool_calls:
            return "end"
        if state["tool_rounds"] >= MAX_TOOL_ROUNDS:
            return "limit"
        return "tools"

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("mcp_tools", mcp_tools_node)
    builder.add_node("limit", limit_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "mcp_tools", "limit": "limit", "end": END},
    )
    builder.add_edge("mcp_tools", "model")
    builder.add_edge("limit", END)
    return builder.compile()
