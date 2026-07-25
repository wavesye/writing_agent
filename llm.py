"""DeepSeek 多工具自动选择与循环调度。"""

import json
import os

from openai import OpenAI

from tools import TOOL_HANDLERS

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
MAX_TOOL_ROUNDS = 5

VIN_PARAMETER = {
    "type": "object",
    "properties": {
        "vin": {
            "type": "string",
            "description": "车辆识别码，例如 VIN123",
        }
    },
    "required": ["vin"],
    "additionalProperties": False,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_vin",
            "description": "分析 VIN，返回状态、温度（摄氏度）和异常标记。",
            "parameters": VIN_PARAMETER,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_info",
            "description": "根据 VIN 查询车辆品牌、车型和年份。",
            "parameters": VIN_PARAMETER,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_advice",
            "description": (
                "根据温度和异常标记获取维修建议。参数应来自 analyze_vin "
                "的结果；缺少指标时先调用 analyze_vin。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "number",
                        "description": "车辆温度，单位为摄氏度",
                    },
                    "has_anomaly": {
                        "type": "boolean",
                        "description": "是否发现异常",
                    },
                },
                "required": ["temperature", "has_anomaly"],
                "additionalProperties": False,
            },
        },
    },
]


def run_agent(user_input: str) -> str:
    """让模型自动选择并按需连续调用多个工具。"""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是车辆助手。根据用户目标自主选择必要工具，不调用无关工具。"
                "维修建议必须基于 analyze_vin 返回的真实参数。"
                "所有工具数据均为演示数据，最终用中文简洁说明这一点。"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or ""

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

    raise RuntimeError(f"工具调用超过最大轮数 {MAX_TOOL_ROUNDS}")
