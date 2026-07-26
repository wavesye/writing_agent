"""DeepSeek Provider。"""

import os

from openai import OpenAI


class DeepSeekProvider:
    name = "deepseek"

    def __init__(self):
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        result = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [
                call.model_dump(exclude_none=True)
                for call in message.tool_calls
            ]
        return result
