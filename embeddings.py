"""Configurable embedding clients for local and cloud vector RAG."""

import os

import requests
from openai import OpenAI


class EmbeddingProvider:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        self.timeout = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
        if self.provider == "ollama":
            self.model = os.getenv("EMBEDDING_MODEL") or os.getenv(
                "OLLAMA_EMBEDDING_MODEL"
            )
            self.base_url = os.getenv(
                "EMBEDDING_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/")
            self.client = None
        else:
            self.model = os.getenv("EMBEDDING_MODEL")
            self.base_url = os.getenv(
                "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
            api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 EMBEDDING_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        if not self.model:
            raise ValueError(
                "请设置 EMBEDDING_MODEL；使用 Ollama 时也可设置 OLLAMA_EMBEDDING_MODEL"
            )

    @property
    def signature(self) -> str:
        return f"{self.provider}:{self.base_url}:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "ollama":
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
