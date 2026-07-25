"""DeepSeek 通过 MCP 动态发现和调用车辆工具。"""

import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_TOOL_ROUNDS = 5
MCP_SERVER = Path(__file__).with_name("mcp_server.py")


def _deepseek_tool(mcp_tool) -> dict:
    """把 MCP Tool Schema 转成 DeepSeek/OpenAI Function Tool Schema。"""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


def _tool_result_text(result) -> str:
    """把 MCP CallToolResult 转成可回传给模型的 JSON 文本。"""
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text"
    ]
    return "\n".join(parts)


async def run_agent(user_input: str) -> str:
    """连接 MCP Server，让模型动态选择并调用其工具。"""
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

            client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是车辆助手，只能使用 MCP 返回的真实数据。"
                        "搜索车队时先调用 list_vehicles，禁止编造 VIN。"
                        "维修建议必须基于 analyze_vin 返回的参数。"
                        "最终用中文简洁回答，并说明数据为演示数据。"
                    ),
                },
                {"role": "user", "content": user_input},
            ]

            for _ in range(MAX_TOOL_ROUNDS):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                message = response.choices[0].message
                messages.append(message)
                if not message.tool_calls:
                    return message.content or ""

                for tool_call in message.tool_calls:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = await session.call_tool(name, arguments)
                    if result.isError:
                        raise RuntimeError(f"MCP Tool {name} 调用失败")
                    output = _tool_result_text(result)
                    print(f"[mcp tool] {name}({arguments}) -> {output}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output,
                        }
                    )

    raise RuntimeError(f"工具调用超过最大轮数 {MAX_TOOL_ROUNDS}")
