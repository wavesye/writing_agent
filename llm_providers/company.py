"""公司内部 HTTP LLM Provider，可适配原生或 Prompt Tool Calling。"""

import json
import os
import uuid
from typing import Any

import requests


def _find_value(data: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            found = _find_value(value, keys)
            if found is not None:
                return found
    return None


def _get_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"响应中不存在字段路径 {path!r}")
        current = current[part]
    return current


class CompanyLLMProvider:
    name = "company"

    def __init__(self):
        self.base_url = os.environ["COMPANY_LLM_BASE_URL"].rstrip("/")
        self.token = os.environ["COMPANY_LLM_TOKEN"]
        self.init_path = os.getenv("COMPANY_LLM_INIT_PATH", "/chat/init")
        self.chat_path = os.environ["COMPANY_LLM_CHAT_PATH"]
        self.session_field = os.getenv("COMPANY_LLM_SESSION_FIELD", "")
        self.request_session_field = os.getenv(
            "COMPANY_LLM_SESSION_REQUEST_FIELD",
            "chat_id",
        )
        self.tool_mode = os.getenv(
            "COMPANY_LLM_TOOL_MODE",
            "native",
        ).lower()
        self.model = os.getenv("COMPANY_LLM_MODEL", "")
        self.timeout = float(os.getenv("COMPANY_LLM_TIMEOUT", "60"))
        self.http = requests.Session()
        self.http.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )
        self.chat_id = self._init_chat()

    def _init_chat(self) -> Any:
        response = self.http.get(
            f"{self.base_url}{self.init_path}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if self.session_field:
            return _get_path(data, self.session_field)
        return _find_value(
            data,
            ("chat_id", "chatId", "session_id", "sessionId", "id"),
        )

    def _prompt_messages(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> list[dict]:
        if self.tool_mode != "prompt":
            return messages
        protocol = {
            "instruction": (
                "需要调用工具时只输出 tool_call JSON；可以最终回答时只输出 "
                "final JSON。不要使用 Markdown 代码块。"
            ),
            "tool_call_format": {
                "type": "tool_call",
                "name": "工具名",
                "arguments": {},
            },
            "final_format": {"type": "final", "content": "最终答案"},
            "tools": tools,
        }
        return [
            {
                "role": "system",
                "content": json.dumps(protocol, ensure_ascii=False),
            },
            *messages,
        ]

    def _request_body(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict:
        body = {
            "messages": self._prompt_messages(messages, tools),
        }
        if self.chat_id is not None:
            body[self.request_session_field] = self.chat_id
        if self.model:
            body["model"] = self.model
        if self.tool_mode == "native":
            body["tools"] = tools
        return body

    def _normalize_tool_calls(self, calls: Any) -> list[dict]:
        if not calls:
            return []
        normalized = []
        for call in calls:
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            normalized.append(
                {
                    "id": call.get("id", f"call_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                }
            )
        return normalized

    def _normalize_response(self, data: dict) -> dict:
        if data.get("choices"):
            message = data["choices"][0].get("message", {})
        elif isinstance(data.get("data"), dict):
            message = data["data"].get("message", data["data"])
        elif isinstance(data.get("result"), dict):
            message = data["result"].get("message", data["result"])
        else:
            message = data.get("message", data)

        if isinstance(message, str):
            content = message
            calls = []
        else:
            content = (
                message.get("content")
                or message.get("text")
                or message.get("answer")
                or ""
            )
            calls = message.get("tool_calls") or message.get("toolCalls")

        if self.tool_mode == "prompt":
            parsed = json.loads(content)
            if parsed.get("type") == "tool_call":
                calls = [
                    {
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    }
                ]
                content = None
            elif parsed.get("type") == "final":
                content = parsed.get("content", "")

        result = {"role": "assistant", "content": content}
        normalized_calls = self._normalize_tool_calls(calls)
        if normalized_calls:
            result["tool_calls"] = normalized_calls
        return result

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self.http.post(
            f"{self.base_url}{self.chat_path}",
            json=self._request_body(messages, tools),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._normalize_response(response.json())
