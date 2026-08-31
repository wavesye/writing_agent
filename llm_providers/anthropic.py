"""Native Anthropic Messages API adapter."""

import json
import os
import uuid

import requests


class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.model = os.environ["ANTHROPIC_MODEL"]
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "120"))

    def _messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts, converted, call_names = [], [], {}
        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append(message.get("content") or "")
            elif role == "assistant":
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    call_names[call["id"]] = function["name"]
                    blocks.append({"type": "tool_use", "id": call["id"],
                                   "name": function["name"],
                                   "input": json.loads(function["arguments"])})
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result", "tool_use_id": message["tool_call_id"],
                    "content": message.get("content") or ""
                }
                if (converted and converted[-1]["role"] == "user"
                        and isinstance(converted[-1]["content"], list)
                        and converted[-1]["content"]
                        and converted[-1]["content"][0].get("type") == "tool_result"):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                converted.append({"role": "user", "content": message.get("content") or ""})
        return "\n".join(system_parts), converted

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        system, converted = self._messages(messages)
        body = {"model": self.model, "max_tokens": 4096, "messages": converted}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [{"name": item["function"]["name"],
                              "description": item["function"].get("description", ""),
                              "input_schema": item["function"]["parameters"]}
                             for item in tools]
        response = requests.post(
            f"{self.base_url}/v1/messages", json=body, timeout=self.timeout,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
        )
        response.raise_for_status()
        blocks = response.json().get("content", [])
        text = "\n".join(block.get("text", "") for block in blocks
                         if block.get("type") == "text") or None
        calls = [{"id": block.get("id", f"call_{uuid.uuid4().hex}"),
                  "type": "function", "function": {"name": block["name"],
                  "arguments": json.dumps(block.get("input", {}), ensure_ascii=False)}}
                 for block in blocks if block.get("type") == "tool_use"]
        result = {"role": "assistant", "content": text}
        if calls:
            result["tool_calls"] = calls
        return result
