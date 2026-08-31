"""Expose the local academic writing corpus as MCP tools."""

from mcp.server.fastmcp import FastMCP
from knowledge_base import KnowledgeBase

mcp = FastMCP("Academic Writing Tools")
knowledge_base = KnowledgeBase()


@mcp.tool()
def search_style_corpus(query: str, top_k: int = 5) -> dict:
    """Hybrid semantic/vector and keyword search over reference-paper prose."""
    return knowledge_base.search(query, top_k)


@mcp.tool()
def list_corpus_sources() -> dict:
    """List papers currently indexed in the local writing-style corpus."""
    return knowledge_base.list_sources()


if __name__ == "__main__":
    mcp.run(transport="stdio")
