"""FastMCP server exposing required tools (stdio or streamable-http via FASTMCP_* env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from mcp_server import tool_runtime as tr

# Read host/port explicitly so Docker environment variables are always honoured,
# regardless of when pydantic-settings resolves the FASTMCP_* prefix.
_HOST = os.getenv("FASTMCP_HOST", "0.0.0.0")
_PORT = int(os.getenv("FASTMCP_PORT", "8765"))

mcp = FastMCP(
    "Agentic Social Media Post Studio",
    instructions="Tools for web search, URL fetch, PDF indexing/query, and source listing.",
    host=_HOST,
    port=_PORT,
)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """Run a search query and return ranked results with title, url, snippet."""
    return tr.web_search(query, max_results=max_results)


@mcp.tool()
def fetch_url(url: str) -> dict:
    """Fetch a URL, strip boilerplate, return markdown plus extracted image URLs and web_source_id."""
    return tr.fetch_url(url)


@mcp.tool()
def pdf_query(pdf_id: str, question: str, k: int = 6) -> dict:
    """Retrieve top-k chunks for an indexed pdf_id with page/modality metadata."""
    return tr.pdf_query(pdf_id, question, k=k)


@mcp.tool()
def index_pdf(file_path: str) -> dict:
    """Ingest a PDF from disk (must be under storage/uploads), return pdf_id."""
    return tr.index_pdf(file_path)


@mcp.tool()
def list_sources() -> dict:
    """Enumerate indexed chunks and distinct pdf_ids in the knowledge base."""
    return tr.list_sources()


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
