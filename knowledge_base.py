"""Academic corpus ingestion with persistent vector + FTS5 hybrid retrieval."""

import hashlib
import os
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DEFAULT_DB_PATH = BASE_DIR / "data" / "knowledge.sqlite"
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
                 db_path: Path = DEFAULT_DB_PATH, chunk_size: int = 1400,
                 chunk_overlap: int = 180, rag_mode: str | None = None,
                 embedder=None, vector_store=None):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rag_mode = (rag_mode or os.getenv("RAG_MODE", "hybrid")).lower()
        if self.rag_mode not in {"hybrid", "vector", "keyword"}:
            raise ValueError("RAG_MODE 必须是 hybrid、vector 或 keyword")
        self.embedder = embedder
        self.vector_store = vector_store

    def _files(self) -> list[Path]:
        if not self.knowledge_dir.exists():
            return []
        return sorted(p for p in self.knowledge_dir.rglob("*")
                      if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.chunk_size}:{self.chunk_overlap}".encode())
        for path in files:
            digest.update(str(path.relative_to(self.knowledge_dir)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _split(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                candidates = [text.rfind(mark, start + self.chunk_size // 2, end)
                              for mark in ("\n\n", "。", ". ", "; ")]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def _document_chunks(self, path: Path) -> list[dict]:
        source = str(path.relative_to(self.knowledge_dir))
        records = []
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as error:
                raise RuntimeError(
                    "读取 PDF 需要 pypdf；请先运行 pip install -r requirements.txt"
                ) from error
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, 1):
                for index, content in enumerate(self._split(page.extract_text() or ""), 1):
                    records.append({"source": source, "location": f"p.{page_number}",
                                    "chunk_no": index, "content": content})
            return records

        text = path.read_text(encoding="utf-8")
        section, buffer = path.stem, []

        def flush():
            content = "\n".join(buffer).strip()
            for index, chunk in enumerate(self._split(content), 1):
                records.append({"source": source, "location": section,
                                "chunk_no": index, "content": chunk})
            buffer.clear()

        for line in text.splitlines():
            if path.suffix.lower() == ".md" and line.startswith("#"):
                flush()
                section = line.lstrip("#").strip() or path.stem
            else:
                buffer.append(line)
        flush()
        return records

    def ensure_index(self) -> int:
        files = self._files()
        if not files:
            raise RuntimeError(
                f"知识库为空。请将 PDF、Markdown 或 TXT 放入：{self.knowledge_dir}"
            )
        fingerprint = self._fingerprint(files)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS kb_meta "
                         "(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            current = conn.execute(
                "SELECT value FROM kb_meta WHERE key='fingerprint'"
            ).fetchone()
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='knowledge_chunks'"
            ).fetchone()
            if not (current and current[0] == fingerprint and table_exists):
                conn.execute("DROP TABLE IF EXISTS knowledge_chunks")
                conn.execute("CREATE VIRTUAL TABLE knowledge_chunks USING fts5("
                             "source UNINDEXED, location UNINDEXED, chunk_no UNINDEXED, "
                             "content, tokenize='trigram')")
                chunks = [chunk for path in files for chunk in self._document_chunks(path)]
                if not chunks:
                    raise RuntimeError("未能从知识库文件中提取文本；扫描版 PDF 请先进行 OCR。")
                conn.executemany(
                    "INSERT INTO knowledge_chunks(source, location, chunk_no, content) "
                    "VALUES (:source, :location, :chunk_no, :content)", chunks)
                conn.execute("INSERT OR REPLACE INTO kb_meta(key, value) "
                             "VALUES ('fingerprint', ?)", (fingerprint,))
                conn.commit()
            rows = conn.execute(
                "SELECT source, location, chunk_no, content FROM knowledge_chunks"
            ).fetchall()
        chunks = [{"source": source, "location": location, "chunk_no": chunk_no,
                   "content": content} for source, location, chunk_no, content in rows]
        if self.rag_mode != "keyword":
            if self.embedder is None:
                from embeddings import EmbeddingProvider
                self.embedder = EmbeddingProvider()
            if self.vector_store is None:
                from vector_store import ChromaVectorStore
                self.vector_store = ChromaVectorStore(
                    self.db_path.with_name("chroma"), self.embedder
                )
            vector_fingerprint = hashlib.sha256(
                f"{fingerprint}:{self.embedder.signature}".encode()
            ).hexdigest()
            self.vector_store.ensure_index(chunks, vector_fingerprint)
        return len(chunks)

    def _match_expression(self, query: str) -> str:
        compact = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "", query)
        terms = [compact[i:i + 3] for i in range(max(0, len(compact) - 2))]
        terms.extend(re.findall(r"[A-Za-z0-9_-]{3,}", query))
        unique = list(dict.fromkeys(terms))[:48]
        if not unique:
            raise ValueError("检索问题不能为空")
        return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"'
                           for term in unique)

    def search(self, query: str, top_k: int = 5) -> dict:
        chunk_count = self.ensure_index()
        top_k = max(1, min(top_k, 10))
        candidate_count = min(30, top_k * 3)
        lexical = []
        if self.rag_mode in {"hybrid", "keyword"}:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT source, location, chunk_no, content, "
                    "bm25(knowledge_chunks) FROM knowledge_chunks "
                    "WHERE knowledge_chunks MATCH ? "
                    "ORDER BY bm25(knowledge_chunks) LIMIT ?",
                    (self._match_expression(query), candidate_count),
                ).fetchall()
            lexical = [{"source": source, "location": location,
                        "chunk_no": chunk_no, "content": content,
                        "keyword_rank": rank}
                       for rank, (source, location, chunk_no, content, _)
                       in enumerate(rows, 1)]
        vector = []
        if self.rag_mode in {"hybrid", "vector"}:
            vector = self.vector_store.search(query, candidate_count)

        fused = {}
        for weight, rank_field, candidates in (
            (0.4, "keyword_rank", lexical), (0.6, "vector_rank", vector)
        ):
            for item in candidates:
                key = (item["source"], item["location"], int(item["chunk_no"]))
                entry = fused.setdefault(key, dict(item, score=0.0))
                entry.update(item)
                entry["score"] += weight / (60 + item[rank_field])
        ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        results = []
        for rank, item in enumerate(ranked, 1):
            item = dict(item)
            item.update({"rank": rank,
                         "citation": f"[{item['source']}#{item['location']}]"})
            results.append(item)
        return {"query": query, "indexed_chunks": chunk_count,
                "retrieval_mode": self.rag_mode, "results": results}

    def list_sources(self) -> dict:
        count = self.ensure_index()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT source, COUNT(*) FROM knowledge_chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return {"indexed_chunks": count,
                "sources": [{"source": source, "chunks": chunks}
                            for source, chunks in rows]}


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(f"已索引 {kb.ensure_index()} 个片段：{DEFAULT_DB_PATH}")
