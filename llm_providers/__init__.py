"""Configurable provider factory for common cloud and local LLMs."""

import os

from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider():
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider in {"anthropic", "claude"}:
        return AnthropicProvider()
    if provider in {"google", "gemini"}:
        return GeminiProvider()
    return OpenAICompatibleProvider(provider)
