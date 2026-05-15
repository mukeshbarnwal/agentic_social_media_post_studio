#!/usr/bin/env python3
"""Smoke-test MCP tool implementations without starting the FastMCP HTTP server."""

from __future__ import annotations

from mcp_server.tool_runtime import list_sources, web_search


def main() -> None:
    print("list_sources:", list_sources())
    print("web_search:", web_search("MOCK smoke query", max_results=2))


if __name__ == "__main__":
    main()
