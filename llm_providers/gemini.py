"""Native Google Gemini generateContent API adapter."""

import json
import os
import uuid

import requests


class GeminiProvider:
    name = "gemini"

    def __init__(self):
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = os.environ["GEMINI_MODEL"]
        self.base_url = os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "120"))

    def _contents(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts, contents, call_names = [], [], {}
        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append(message.get("content") or "")
                continue
            if role == "assistant":
                parts = []
                if message.get("content"):
                    parts.append({"text": message["content"]})
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    call_names[call["id"]] = function["name"]
                    parts.append({"functionCall": {"name": function["name"],
                                  "args": json.loads(function["arguments"])}})
                contents.append({"role": "model", "parts": parts})
            elif role == "tool":
                part = {"functionResponse": {
                    "name": call_names.get(message["tool_call_id"], "tool"),
                    "response": {"result": message.get("content") or ""}
                }}
                if (contents and contents[-1]["role"] == "user"
                        and "functionResponse" in contents[-1]["parts"][0]):
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "user", "parts": [part]})
            else:
                contents.append({"role": "user", "parts": [{"text": message.get("content") or ""}]})
        return "\n".join(system_parts), contents

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        system, contents = self._contents(messages)
        body = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [{
                "name": item["function"]["name"],
                "description": item["function"].get("description", ""),
                "parameters": item["function"]["parameters"],
            } for item in tools]}]
        response = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key}, json=body, timeout=self.timeout,
        )
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"].get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if "text" in part) or None
        calls = [{"id": f"call_{uuid.uuid4().hex}", "type": "function",
                  "function": {"name": part["functionCall"]["name"],
                  "arguments": json.dumps(part["functionCall"].get("args", {}),
                                          ensure_ascii=False)}}
                 for part in parts if "functionCall" in part]
        result = {"role": "assistant", "content": text}
        if calls:
            result["tool_calls"] = calls
        return result
