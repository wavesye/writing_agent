"""连接 MCP，并通过 LangGraph 运行车辆 Agent。"""

import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent_graph import build_agent_graph
from llm_providers import create_provider

MCP_SERVER = Path(__file__).with_name("mcp_server.py")


def _deepseek_tool(mcp_tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


async def run_agent(user_input: str) -> str:
    """建立 MCP 会话，编译并执行一次 LangGraph。"""
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            discovered = await session.list_tools()
            tools = [_deepseek_tool(tool) for tool in discovered.tools]
            print(f"[mcp] 已发现工具: {[tool.name for tool in discovered.tools]}")

            provider = create_provider()
            print(f"[llm] 当前 Provider: {provider.name}")
            graph = build_agent_graph(provider, session, tools)
            state = await graph.ainvoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是车辆助手，只能使用 MCP 返回的真实数据。"
                                "搜索车队时先调用 list_vehicles，禁止编造 VIN。"
                                "维修建议必须基于 analyze_vin 返回的参数。"
                                "对于解释、SQL 编写和编程问题，可以直接回答，不需要调用工具。"
                                "最终用中文简洁回答，并说明数据为演示数据。"
                            ),
                        },
                        {"role": "user", "content": user_input},
                    ],
                    "tool_rounds": 0,
                    "phase": "start",
                    "tool_trace": [],
                    "final_answer": "",
                },
                {"recursion_limit": 20},
            )
            print(
                f"[graph] 完成 phase={state['phase']}, "
                f"tool_rounds={state['tool_rounds']}, "
                f"trace={state['tool_trace']}"
            )
            return state["final_answer"]
