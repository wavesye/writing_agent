"""Provider configuration and message-conversion tests; no network calls."""

import os
import unittest
from unittest.mock import patch

from llm_providers.anthropic import AnthropicProvider
from llm_providers.gemini import GeminiProvider
from llm_providers.openai_compatible import OpenAICompatibleProvider


class ProviderTest(unittest.TestCase):
    @patch("llm_providers.openai_compatible.OpenAI")
    def test_openai_compatible_custom_configuration(self, client):
        environment = {"LLM_BASE_URL": "http://localhost:9000/v1",
                       "LLM_API_KEY": "test", "LLM_MODEL": "local-model"}
        with patch.dict(os.environ, environment, clear=True):
            provider = OpenAICompatibleProvider("custom")
        client.assert_called_once_with(api_key="test",
                                       base_url="http://localhost:9000/v1")
        self.assertEqual(provider.model, "local-model")

    @patch("llm_providers.openai_compatible.OpenAI")
    def test_ollama_does_not_require_key(self, client):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "qwen"}, clear=True):
            OpenAICompatibleProvider("ollama")
        client.assert_called_once_with(api_key="ollama",
                                       base_url="http://127.0.0.1:11434/v1")

    def test_anthropic_tool_messages(self):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        system, messages = provider._messages([
            {"role": "system", "content": "write academically"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "function": {"name": "search_style_corpus",
                "arguments": '{"query":"methods"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "tool", "tool_call_id": "call-1", "content": "result 2"},
        ])
        self.assertEqual(system, "write academically")
        self.assertEqual(messages[0]["content"][0]["type"], "tool_use")
        self.assertEqual(messages[1]["content"][0]["type"], "tool_result")
        self.assertEqual(len(messages[1]["content"]), 2)

    def test_gemini_tool_messages(self):
        provider = GeminiProvider.__new__(GeminiProvider)
        system, messages = provider._contents([
            {"role": "system", "content": "write academically"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "function": {"name": "search_style_corpus",
                "arguments": '{"query":"methods"}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "tool", "tool_call_id": "call-1", "content": "result 2"},
        ])
        self.assertEqual(system, "write academically")
        self.assertEqual(messages[0]["role"], "model")
        self.assertEqual(messages[1]["parts"][0]["functionResponse"]["name"],
                         "search_style_corpus")
        self.assertEqual(len(messages[1]["parts"]), 2)


if __name__ == "__main__":
    unittest.main()
