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
    conversation_summary: NotRequired[str]
    summary_cursor: NotRequired[int]
    summary_count: NotRequired[int]


def build_agent_graph(
    provider,
    mcp_session,
    tools,
    checkpointer=None,
    summary_trigger: int = 20,
    recent_message_count: int = 8,
):
    """注入模型与 MCP 会话，然后编译可执行状态图。"""
    if summary_trigger <= recent_message_count:
        raise ValueError(
            "summary_trigger 必须大于 recent_message_count，"
            "否则无法稳定压缩上下文"
        )

    def check_context_node(state: AgentState) -> dict:
        unsummarized = len(state["messages"]) - state.get(
            "summary_cursor",
            0,
        )
        print(
            f"[graph:check_context] 未摘要消息={unsummarized}, "
            f"触发阈值={summary_trigger}"
        )
        return {"phase": "check_context"}

    def route_context(state: AgentState) -> Literal["summarize", "model"]:
        unsummarized = len(state["messages"]) - state.get(
            "summary_cursor",
            0,
        )
        if unsummarized > summary_trigger:
            return "summarize"
        return "model"

    def summarize_node(state: AgentState) -> dict:
        cursor = state.get("summary_cursor", 0)
        cutoff = max(cursor, len(state["messages"]) - recent_message_count)
        messages_to_summarize = state["messages"][cursor:cutoff]
        if not messages_to_summarize:
            return {"phase": "summarize"}

        previous = state.get("conversation_summary", "")
        summary_prompt = (
            "请更新论文写作 Agent 的历史摘要。只保留对后续任务有用的信息："
            "研究主题、目标期刊或读者、术语、文风偏好、已确认修改和待处理动作。"
            "区分已确认事实与模型建议，不要编造，不要输出开场白。\n\n"
            f"已有摘要：\n{previous or '无'}\n\n"
            "新增历史消息：\n"
            + json.dumps(messages_to_summarize, ensure_ascii=False)
        )
        reply = provider.generate(
            [
                {
                    "role": "system",
                    "content": "你是精确的会话摘要器。",
                },
                {"role": "user", "content": summary_prompt},
            ],
            [],
        )
        summary = reply.get("content") or previous
        print(
            f"[graph:summarize] 已摘要 {len(messages_to_summarize)} 条，"
            f"保留最近 {len(state['messages']) - cutoff} 条"
        )
        return {
            "conversation_summary": summary,
            "summary_cursor": cutoff,
            "summary_count": state.get("summary_count", 0) + 1,
            "phase": "summarize",
        }

    def model_node(state: AgentState) -> dict:
        print(f"[graph:model] 第 {state['tool_rounds'] + 1} 次决策")
        system_content = (
            "你是严谨的学术论文写作与润色助手。优先保持作者原意、论证逻辑、"
            "专业术语、数字、公式和已有引文不变；不得补造事实、实验结果或参考文献。"
            "当用户要求润色、仿写、改写文风或生成论文段落时，必须先调用 "
            "search_style_corpus，以主题、章节功能和关键术语构造检索词。"
            "学习检索片段的句长、衔接方式、语气、信息密度和段落结构，但不得复制"
            "其独特表达或连续长句。若检索结果不足，应明确说明，不得假装参考过语料。"
            "输出默认包括修改稿；必要时附关键修改说明。知识库只作为文风样本，"
            "不是事实依据；仅在解释所参考的风格样本时使用[文件名#页码/章节]标注。"
            "用户只要求校对时，做最小修改；用户未指定语言时沿用原文语言。"
        )
        summary = state.get("conversation_summary")
        if summary:
            system_content += f"\n历史会话摘要：\n{summary}"
        recent_messages = state["messages"][
            state.get("summary_cursor", 0):
        ]
        message = provider.generate(
            [
                {"role": "system", "content": system_content},
                *recent_messages,
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
    builder.add_node("check_context", check_context_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("model", model_node)
    builder.add_node("mcp_tools", mcp_tools_node)
    builder.add_node("limit", limit_node)
    builder.add_edge(START, "check_context")
    builder.add_conditional_edges(
        "check_context",
        route_context,
        {"summarize": "summarize", "model": "model"},
    )
    builder.add_edge("summarize", "model")
    builder.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "mcp_tools", "limit": "limit", "end": END},
    )
    builder.add_edge("mcp_tools", "check_context")
    builder.add_edge("limit", END)
    return builder.compile(checkpointer=checkpointer)
