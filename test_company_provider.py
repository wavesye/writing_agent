"""公司模型响应格式转换测试，不访问公司网络。"""

import unittest

from llm_providers.company import CompanyLLMProvider


class CompanyProviderTest(unittest.TestCase):
    def provider(self, mode="native"):
        provider = CompanyLLMProvider.__new__(CompanyLLMProvider)
        provider.tool_mode = mode
        return provider

    def test_openai_style_tool_call(self):
        result = self.provider()._normalize_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "analyze_vin",
                                        "arguments": '{"vin":"VIN123"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        self.assertEqual(
            result["tool_calls"][0]["function"]["name"],
            "analyze_vin",
        )

    def test_prompt_style_tool_call(self):
        result = self.provider("prompt")._normalize_response(
            {
                "data": {
                    "content": (
                        '{"type":"tool_call","name":"list_vehicles",'
                        '"arguments":{}}'
                    )
                }
            }
        )
        self.assertEqual(
            result["tool_calls"][0]["function"]["name"],
            "list_vehicles",
        )

    def test_plain_answer(self):
        result = self.provider()._normalize_response(
            {"data": {"answer": "你好"}}
        )
        self.assertEqual(result["content"], "你好")


if __name__ == "__main__":
    unittest.main()
