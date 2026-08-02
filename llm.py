"""连接 MCP，并用 SQLite Checkpointer 运行有记忆的 LangGraph。"""

import os
import sys
from pathlib import Path

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent_graph import build_agent_graph
from llm_providers import create_provider

MCP_SERVER = Path(__file__).with_name("mcp_server.py")
CHECKPOINT_DB = Path(
    os.getenv(
        "AGENT_CHECKPOINT_DB",
        Path(__file__).with_name("data") / "checkpoints.sqlite",
    )
)
SUMMARY_TRIGGER = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "20"))
SUMMARY_KEEP_RECENT = int(os.getenv("SUMMARY_KEEP_RECENT", "8"))


def _deepseek_tool(mcp_tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


async def run_agent(user_input: str, thread_id: str | None = None) -> str:
    """在指定 thread_id 中恢复状态并执行一轮 Agent。"""
    active_thread = thread_id or os.getenv(
        "AGENT_THREAD_ID",
        "vin-agent-cli",
    )
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
    )
    async with AsyncSqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                discovered = await session.list_tools()
                tools = [_deepseek_tool(tool) for tool in discovered.tools]
                print(
                    f"[mcp] 已发现工具: "
                    f"{[tool.name for tool in discovered.tools]}"
                )

                provider = create_provider()
                print(f"[llm] 当前 Provider: {provider.name}")
                print(f"[memory] thread_id={active_thread}")
                graph = build_agent_graph(
                    provider,
                    session,
                    tools,
                    checkpointer=checkpointer,
                    summary_trigger=SUMMARY_TRIGGER,
                    recent_message_count=SUMMARY_KEEP_RECENT,
                )
                state = await graph.ainvoke(
                    {
                        "messages": [
                            {"role": "user", "content": user_input}
                        ],
                        "tool_rounds": 0,
                        "phase": "start",
                        "tool_trace": [],
                        "final_answer": "",
                    },
                    {
                        "configurable": {"thread_id": active_thread},
                        "recursion_limit": 20,
                    },
                )
                print(
                    f"[graph] 完成 phase={state['phase']}, "
                    f"tool_rounds={state['tool_rounds']}, "
                    f"current_vin={state.get('current_vin')}, "
                    f"summaries={state.get('summary_count', 0)}, "
                    f"trace={state['tool_trace']}"
                )
                return state["final_answer"]
