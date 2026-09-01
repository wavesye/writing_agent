"""Tests for local academic-style corpus indexing."""

import tempfile
import unittest
from pathlib import Path

from knowledge_base import KnowledgeBase


class FakeEmbedder:
    signature = "fake:v1"

    def embed(self, texts):
        return [[1.0, 0.0] if ("cat" in text.lower() or "feline" in text.lower())
                else [0.0, 1.0] for text in texts]


class KnowledgeBaseTest(unittest.TestCase):
    def test_nested_markdown_search_and_citation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "knowledge" / "journal"
            corpus.mkdir(parents=True)
            (corpus / "paper.md").write_text(
                "# Introduction\n\nPrior studies establish a robust theoretical foundation.\n"
                "## Method\n\nWe estimate the model using longitudinal observations.",
                encoding="utf-8",
            )
            kb = KnowledgeBase(root / "knowledge", root / "knowledge.sqlite",
                               rag_mode="keyword")
            self.assertEqual(kb.ensure_index(), 2)
            result = kb.search("longitudinal model estimation", 1)
            self.assertEqual(result["results"][0]["source"], "journal/paper.md")
            self.assertEqual(result["results"][0]["location"], "Method")
            self.assertEqual(result["results"][0]["citation"],
                             "[journal/paper.md#Method]")

    def test_document_change_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "knowledge"
            corpus.mkdir()
            document = corpus / "paper.md"
            document.write_text("# Abstract\n\nFirst version.", encoding="utf-8")
            kb = KnowledgeBase(corpus, root / "knowledge.sqlite", rag_mode="keyword")
            self.assertEqual(kb.ensure_index(), 1)
            document.write_text(
                "# Abstract\n\nFirst section.\n# Results\n\nSecond section.",
                encoding="utf-8",
            )
            self.assertEqual(kb.ensure_index(), 2)
            self.assertEqual(len(kb.list_sources()["sources"]), 1)

    def test_listing_sources_does_not_initialize_vector_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "knowledge"
            corpus.mkdir()
            (corpus / "paper.txt").write_text("Academic prose.", encoding="utf-8")

            class FailingVectorStore:
                def ensure_index(self, *_args):
                    raise AssertionError("list_sources 不应初始化向量库")

            kb = KnowledgeBase(corpus, root / "knowledge.sqlite", rag_mode="hybrid",
                               embedder=FakeEmbedder(), vector_store=FailingVectorStore())
            self.assertEqual(kb.list_sources()["sources"][0]["source"], "paper.txt")

    def test_empty_corpus_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "knowledge").mkdir()
            kb = KnowledgeBase(root / "knowledge", root / "knowledge.sqlite",
                               rag_mode="keyword")
            with self.assertRaisesRegex(RuntimeError, "PDF"):
                kb.ensure_index()

    def test_hybrid_rag_finds_semantic_match_in_chroma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "knowledge"
            corpus.mkdir()
            (corpus / "paper.md").write_text(
                "# Relevant\n\nFeline behavior is examined longitudinally.\n"
                "# Other\n\nMacroeconomic policy affects aggregate demand.",
                encoding="utf-8",
            )
            kb = KnowledgeBase(corpus, root / "knowledge.sqlite",
                               rag_mode="hybrid", embedder=FakeEmbedder())
            result = kb.search("cat", 1)
            self.assertEqual(result["retrieval_mode"], "hybrid")
            self.assertEqual(result["results"][0]["location"], "Relevant")
            self.assertEqual(result["results"][0]["vector_rank"], 1)


if __name__ == "__main__":
    unittest.main()
