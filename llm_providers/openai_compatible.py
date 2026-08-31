"""Provider for OpenAI and services implementing the Chat Completions API."""

import os

from openai import OpenAI


PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "OPENAI_MODEL"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL"),
    "ollama": ("http://127.0.0.1:11434/v1", "OLLAMA_API_KEY", "OLLAMA_MODEL"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "QWEN_MODEL"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "MOONSHOT_MODEL"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "ZHIPU_MODEL"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY", "SILICONFLOW_MODEL"),
}


class OpenAICompatibleProvider:
    def __init__(self, provider: str):
        self.name = provider
        preset = PRESETS.get(provider)
        if preset:
            default_url, key_variable, model_variable = preset
            base_url = os.getenv("LLM_BASE_URL", default_url)
            api_key = os.getenv("LLM_API_KEY") or os.getenv(key_variable)
            model = os.getenv("LLM_MODEL") or os.getenv(model_variable)
        else:
            base_url = os.getenv("LLM_BASE_URL")
            api_key = os.getenv("LLM_API_KEY")
            model = os.getenv("LLM_MODEL")
        if not base_url:
            raise ValueError("请设置 LLM_BASE_URL（OpenAI-compatible API 地址）")
        if not model:
            raise ValueError("请设置 LLM_MODEL，或所选 Provider 对应的模型变量")
        if not api_key and provider != "ollama":
            raise ValueError("请设置 LLM_API_KEY，或所选 Provider 对应的 API Key")
        self.model = model
        self.client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

    def generate(self, messages: list[dict], tools: list[dict]) -> dict:
        request = {"model": self.model, "messages": messages}
        if tools:
            request["tools"] = tools
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        result = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = [call.model_dump(exclude_none=True)
                                    for call in message.tool_calls]
        return result
