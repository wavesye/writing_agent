"""本地 Markdown 知识库：分段、SQLite FTS5 索引与检索。"""

import hashlib
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DEFAULT_DB_PATH = BASE_DIR / "data" / "knowledge.sqlite"


class KnowledgeBase:
    def __init__(
        self,
        knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR,
        db_path: Path = DEFAULT_DB_PATH,
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.db_path = Path(db_path)

    def _files(self) -> list[Path]:
        return sorted(self.knowledge_dir.glob("*.md"))

    def _fingerprint(self, files: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _chunks(self, path: Path) -> list[dict]:
        chunks = []
        section = path.stem
        buffer = []

        def flush() -> None:
            content = "\n".join(buffer).strip()
            if content:
                chunks.append(
                    {
                        "source": path.name,
                        "section": section,
                        "content": content,
                    }
                )
            buffer.clear()

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                flush()
                section = line.lstrip("#").strip() or path.stem
            else:
                buffer.append(line)
        flush()
        return chunks

    def ensure_index(self) -> int:
        """文档变化时重建索引，返回当前片段数。"""
        files = self._files()
        if not files:
            raise RuntimeError(f"知识库中没有 Markdown：{self.knowledge_dir}")
        fingerprint = self._fingerprint(files)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS kb_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            current = conn.execute(
                "SELECT value FROM kb_meta WHERE key='fingerprint'"
            ).fetchone()
            if current and current[0] == fingerprint:
                row = conn.execute(
                    "SELECT COUNT(*) FROM knowledge_chunks"
                ).fetchone()
                return int(row[0])

            conn.execute("DROP TABLE IF EXISTS knowledge_chunks")
            conn.execute(
                "CREATE VIRTUAL TABLE knowledge_chunks USING fts5("
                "source UNINDEXED, section, content, tokenize='trigram')"
            )
            chunks = [chunk for path in files for chunk in self._chunks(path)]
            conn.executemany(
                "INSERT INTO knowledge_chunks(source, section, content) "
                "VALUES (:source, :section, :content)",
                chunks,
            )
            conn.execute(
                "INSERT OR REPLACE INTO kb_meta(key, value) "
                "VALUES ('fingerprint', ?)",
                (fingerprint,),
            )
            conn.commit()
            return len(chunks)

    def _expand_query(self, query: str) -> str:
        """为演示领域补充少量同义词；向量检索阶段可移除。"""
        additions = []
        temperatures = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:℃|度)?", query)
        ]
        if temperatures and max(temperatures) >= 80:
            additions.append("电池高温紧急处置 停止运行 严重异常")
        elif temperatures and max(temperatures) >= 60:
            additions.append("温度偏高处置 冷却系统 警告状态")
        if "高温" in query:
            additions.append("电池高温紧急处置 电池温度超过80")
        if "C级" in query.upper():
            additions.append("C级状态 严重异常 紧急维修")
        return " ".join([query, *additions])

    def _match_expression(self, query: str) -> str:
        expanded = self._expand_query(query)
        compact = re.sub(
            r"[^A-Za-z0-9_\-\u4e00-\u9fff]+",
            "",
            expanded,
        )
        terms = [
            compact[index:index + 3]
            for index in range(max(0, len(compact) - 2))
        ]
        terms.extend(re.findall(r"[A-Za-z0-9_-]{3,}", expanded))
        unique = list(dict.fromkeys(terms))[:32]
        if not unique:
            raise ValueError("检索问题不能为空")
        return " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"'
            for term in unique
        )

    def search(self, query: str, top_k: int = 5) -> dict:
        """返回带来源、章节和片段内容的检索结果。"""
        chunk_count = self.ensure_index()
        top_k = max(1, min(top_k, 10))
        expression = self._match_expression(query)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT source, section, content, "
                "bm25(knowledge_chunks) AS score "
                "FROM knowledge_chunks WHERE knowledge_chunks MATCH ? "
                "ORDER BY score LIMIT ?",
                (expression, max(10, top_k * 4)),
            ).fetchall()
        temperatures = [
            float(value)
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*(?:℃|度)?", query)
        ]

        def domain_priority(row) -> int:
            _, section, content, _ = row
            text = section + content
            priority = 0
            if (
                (temperatures and max(temperatures) >= 80)
                or "高温" in query
            ) and "高温" in text:
                priority += 10
            if "工单" in query and "工单" in section:
                priority += 10
            if "C级" in query.upper() and "C级" in section:
                priority += 10
            return priority

        rows.sort(key=lambda row: (-domain_priority(row), row[3]))
        rows = rows[:top_k]
        results = [
            {
                "rank": index,
                "source": source,
                "section": section,
                "content": content,
            }
            for index, (source, section, content, _) in enumerate(rows, 1)
        ]
        return {
            "query": query,
            "indexed_chunks": chunk_count,
            "results": results,
        }


if __name__ == "__main__":
    count = KnowledgeBase().ensure_index()
    print(f"已索引 {count} 个知识片段：{DEFAULT_DB_PATH}")
