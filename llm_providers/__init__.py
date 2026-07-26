"""模型 Provider 工厂。"""

import os

from .company import CompanyLLMProvider
from .deepseek import DeepSeekProvider


def create_provider():
    name = os.getenv("LLM_PROVIDER", "deepseek").lower()
    if name == "deepseek":
        return DeepSeekProvider()
    if name == "company":
        return CompanyLLMProvider()
    raise ValueError(
        f"不支持的 LLM_PROVIDER={name!r}，可选 deepseek 或 company"
    )
