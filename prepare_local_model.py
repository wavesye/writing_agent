"""Download and verify the in-process ONNX embedding model before packaging."""

from embeddings import EmbeddingProvider


if __name__ == "__main__":
    provider = EmbeddingProvider()
    vector = provider.embed(["Academic writing model readiness check."])[0]
    print(f"本地 embedding 模型准备完成：{provider.model}，维度={len(vector)}")
