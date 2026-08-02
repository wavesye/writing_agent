"""LangGraph 状态图：DeepSeek 决策，MCP 执行工具。"""

import json
import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

MAX_TOOL_ROUNDS = 5


class AgentState(TypedDict):
    """图中每个节点共享和更新的状态。"""

    messages: Annotated[list[dict], operator.add]
    tool_rounds: int
    phase: str
    tool_trace: list[str]
    final_answer: str
    current_vin: NotRequired[str | None]


def build_agent_graph(provider, mcp_session, tools, checkpointer=None):
    """注入模型与 MCP 会话，然后编译可执行状态图。"""

    def model_node(state: AgentState) -> dict:
        print(f"[graph:model] 第 {state['tool_rounds'] + 1} 次决策")
        system_content = (
            "你是车辆助手，只能使用 MCP 返回的真实车辆数据。"
            "搜索车队时先调用 list_vehicles，禁止编造 VIN。"
            "维修建议必须基于 analyze_vin 返回的参数。"
            "对于解释、SQL 编写和编程问题，可以直接回答。"
        )
        current_vin = state.get("current_vin")
        if current_vin:
            system_content += f"当前会话关注的车辆是 {current_vin}。"
        message = provider.generate(
            [
                {"role": "system", "content": system_content},
                *state["messages"],
            ],
            tools,
        )
        return {
            "messages": [message],
            "phase": "model",
            "final_answer": message.get("content") or "",
        }

    async def mcp_tools_node(state: AgentState) -> dict:
        tool_messages = []
        trace = list(state.get("tool_trace", []))
        current_vin = state.get("current_vin")
        for tool_call in state["messages"][-1].get("tool_calls", []):
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            if arguments.get("vin"):
                current_vin = arguments["vin"].upper()
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
            "current_vin": current_vin,
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
    return builder.compile(checkpointer=checkpointer)
