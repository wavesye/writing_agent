"""本地知识库索引与检索测试。"""

import tempfile
import unittest
from pathlib import Path

from knowledge_base import KnowledgeBase


class KnowledgeBaseTest(unittest.TestCase):
    def test_index_search_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            (knowledge / "manual.md").write_text(
                "# 手册\n\n## 高温处置\n\n"
                "电池温度超过80℃时应立即停止运行。\n",
                encoding="utf-8",
            )
            kb = KnowledgeBase(knowledge, root / "knowledge.sqlite")
            self.assertEqual(kb.ensure_index(), 1)
            result = kb.search("电池高温怎么处理", 1)
            self.assertEqual(result["results"][0]["source"], "manual.md")
            self.assertEqual(result["results"][0]["section"], "高温处置")

    def test_document_change_rebuilds_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            document = knowledge / "manual.md"
            document.write_text("# 手册\n\n第一版内容。", encoding="utf-8")
            kb = KnowledgeBase(knowledge, root / "knowledge.sqlite")
            self.assertEqual(kb.ensure_index(), 1)
            document.write_text(
                "# 手册\n\n## 第一节\n\n内容一。\n\n## 第二节\n\n内容二。",
                encoding="utf-8",
            )
            self.assertEqual(kb.ensure_index(), 2)


if __name__ == "__main__":
    unittest.main()
