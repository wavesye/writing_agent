"""In-process local ONNX or cloud embedding clients for vector RAG."""

import os
from pathlib import Path

from openai import OpenAI

from app_paths import BUNDLED_MODEL_DIR, IS_FROZEN, MODEL_DIR, ensure_app_dirs


class EmbeddingProvider:
    def __init__(self):
        self.provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        if self.provider == "local":
            ensure_app_dirs()
            from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
                ONNXMiniLM_L6_V2,
            )
            model_path = os.getenv("EMBEDDING_MODEL_PATH")
            if model_path:
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH = Path(model_path).expanduser().resolve()
            elif IS_FROZEN and BUNDLED_MODEL_DIR.exists():
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH = BUNDLED_MODEL_DIR
            else:
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH = MODEL_DIR / "all-MiniLM-L6-v2"
            self.model = "all-MiniLM-L6-v2"
            self.base_url = "local"
            self.client = ONNXMiniLM_L6_V2()
        elif self.provider in {"openai", "openai-compatible", "cloud"}:
            self.model = os.getenv("EMBEDDING_MODEL")
            self.base_url = os.getenv(
                "EMBEDDING_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/")
            api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("请设置 EMBEDDING_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            if not self.model:
                raise ValueError("使用云端 embedding 时请设置 EMBEDDING_MODEL")
        else:
            raise ValueError("EMBEDDING_PROVIDER 必须是 local 或 openai-compatible")

    @property
    def signature(self) -> str:
        return f"{self.provider}:{self.base_url}:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local":
            return [vector.tolist() for vector in self.client(texts)]
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]
