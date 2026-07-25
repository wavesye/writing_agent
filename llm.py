"""DeepSeek Chat Completions API 与工具定义。"""

import json
import os

from openai import OpenAI

from tools import TOOL_HANDLERS

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_vin",
            "description": (
                "分析车辆 VIN。返回 vin、status、temperature（摄氏度）"
                "和 has_anomaly；当前为本地模拟数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vin": {
                        "type": "string",
                        "description": "车辆识别码，例如 VIN123",
                    }
                },
                "required": ["vin"],
                "additionalProperties": False,
            },
        },
    }
]


def run_agent(user_input: str) -> str:
    """完成 用户 -> LLM -> Tool -> Python -> LLM -> 用户 的闭环。"""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是车辆分析助手。需要 VIN 数据时必须调用工具；"
                "说明工具返回的是演示数据，并用中文简洁回答。"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        extra_body={"thinking": {"type": "disabled"}},
    )
    message = response.choices[0].message
    if not message.tool_calls:
        return message.content or ""

    messages.append(message)
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"未知工具: {name}")

        arguments = json.loads(tool_call.function.arguments)
        result = handler(**arguments)
        print(f"[tool] {name}({arguments}) -> {result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return final_response.choices[0].message.content or ""
