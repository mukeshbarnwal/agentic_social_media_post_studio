"""LangSmith tracing for LangGraph runs and LangChain LLM calls."""

from __future__ import annotations

import os
from typing import Any


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def is_langsmith_enabled() -> bool:
    """True when an API key is set and tracing is explicitly enabled."""
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    return bool(key and key.strip()) and _truthy(tracing)


def configure_langsmith() -> bool:
    """
    Normalize LANGSMITH_* / LANGCHAIN_* env aliases so LangChain auto-traces.

    Call once after loading `.env` (Streamlit, evals, scripts).
    Returns whether tracing is active.
    """
    api_key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)

    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    if tracing:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", tracing)
        os.environ.setdefault("LANGSMITH_TRACING", tracing)

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if project:
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
        os.environ.setdefault("LANGSMITH_PROJECT", project)

    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)

    enabled = is_langsmith_enabled()
    if enabled:
        project_name = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "(default)"
        print(f"[LANGSMITH] Tracing enabled → project={project_name}", flush=True)
    return enabled


def build_invoke_config(
    *,
    run_name: str = "studio_run",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    RunnableConfig for LangGraph `invoke` when LangSmith is on.

    Do not set ``run_id`` here — forcing a custom id can flatten the trace tree in
    LangSmith so ChatOpenAI spans appear as separate top-level runs. Use
    ``metadata.jsonl_run_id`` to correlate with local JSONL instead.
    """
    if not is_langsmith_enabled():
        return {}
    config: dict[str, Any] = {"run_name": run_name}
    if metadata:
        config["metadata"] = metadata
    if tags:
        config["tags"] = tags
    return config
