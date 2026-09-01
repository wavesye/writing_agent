"""连接 MCP，并用 SQLite Checkpointer 运行有记忆的 LangGraph。"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent_graph import build_agent_graph
from llm_providers import create_provider
from app_paths import DATA_DIR, IS_FROZEN, ensure_app_dirs
from knowledge_base import KnowledgeBase

MCP_SERVER = Path(__file__).with_name("mcp_server.py")
CHECKPOINT_DB = Path(
    os.getenv(
        "AGENT_CHECKPOINT_DB",
        DATA_DIR / "checkpoints.sqlite",
    )
)
SUMMARY_TRIGGER = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "20"))
SUMMARY_KEEP_RECENT = int(os.getenv("SUMMARY_KEEP_RECENT", "8"))


def _model_tool(mcp_tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


LOCAL_TOOLS = [
    {"type": "function", "function": {
        "name": "search_style_corpus",
        "description": "Hybrid semantic/vector and keyword search over reference-paper prose.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "list_corpus_sources",
        "description": "List papers currently indexed in the local writing-style corpus.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


class LocalToolSession:
    """In-process tool adapter used by a frozen desktop application."""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()

    async def call_tool(self, name, arguments):
        if name == "search_style_corpus":
            value = self.knowledge_base.search(**arguments)
        elif name == "list_corpus_sources":
            value = self.knowledge_base.list_sources()
        else:
            return SimpleNamespace(isError=True, structuredContent=None, content=[])
        return SimpleNamespace(isError=False, structuredContent=value, content=[])


async def _invoke(provider, session, tools, checkpointer, user_input, active_thread):
    graph = build_agent_graph(
        provider, session, tools, checkpointer=checkpointer,
        summary_trigger=SUMMARY_TRIGGER, recent_message_count=SUMMARY_KEEP_RECENT,
    )
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": user_input}],
         "tool_rounds": 0, "phase": "start", "tool_trace": [], "final_answer": ""},
        {"configurable": {"thread_id": active_thread}, "recursion_limit": 20},
    )
    print(f"[graph] 完成 phase={state['phase']}, tool_rounds={state['tool_rounds']}, "
          f"summaries={state.get('summary_count', 0)}, trace={state['tool_trace']}")
    return state["final_answer"]


async def run_agent(user_input: str, thread_id: str | None = None) -> str:
    """在指定 thread_id 中恢复状态并执行一轮 Agent。"""
    active_thread = thread_id or os.getenv(
        "AGENT_THREAD_ID",
        "writing-agent-cli",
    )
    ensure_app_dirs()
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(
        str(CHECKPOINT_DB)
    ) as checkpointer:
        provider = create_provider()
        print(f"[llm] 当前 Provider: {provider.name}")
        print(f"[memory] thread_id={active_thread}")
        if IS_FROZEN:
            print("[tools] 打包模式：使用进程内知识库工具")
            return await _invoke(provider, LocalToolSession(), LOCAL_TOOLS,
                                 checkpointer, user_input, active_thread)
        server = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER)])
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                discovered = await session.list_tools()
                tools = [_model_tool(tool) for tool in discovered.tools]
                print(
                    f"[mcp] 已发现工具: "
                    f"{[tool.name for tool in discovered.tools]}"
                )

                return await _invoke(provider, session, tools, checkpointer,
                                     user_input, active_thread)
