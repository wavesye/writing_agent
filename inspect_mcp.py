"""连接本地 MCP Server，直观打印它暴露的工具。"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def inspect() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).with_name("mcp_server.py"))],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                print(f"\n{tool.name}\n  {tool.description}")
                print(json.dumps(tool.inputSchema, ensure_ascii=False, indent=2))

            demo = await session.call_tool(
                "search_knowledge",
                {"query": "动力电池高温如何处理", "top_k": 2},
            )
            print("\n调用示例：search_knowledge")
            for item in demo.content:
                if getattr(item, "type", None) == "text":
                    print(item.text)


if __name__ == "__main__":
    asyncio.run(inspect())
