"""Persist user query + pipeline output for offline batch analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.store import project_root


def interactions_dir() -> Path:
    d = project_root() / os.getenv("INTERACTION_LOG_DIR", "storage/interactions")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _jsonl_path() -> Path:
    return interactions_dir() / "interactions.jsonl"


def _enabled() -> bool:
    return os.getenv("INTERACTION_LOG_DISABLED", "").lower() not in ("1", "true", "yes")


def _input_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "topic": state.get("topic"),
        "tone": state.get("tone"),
        "target_length": state.get("target_length"),
        "num_slides": state.get("num_slides"),
        "brand_color": state.get("brand_color"),
        "pdf_ids": list(state.get("pdf_ids") or []),
        "image_paths": list(state.get("image_paths") or []),
        "url_or_query": state.get("url_or_query"),
        "rerun_scope": state.get("rerun_scope", "full"),
        "user_edited_hook": state.get("user_edited_hook"),
        "user_edited_body": state.get("user_edited_body"),
        "image_filenames": [Path(p).name for p in (state.get("image_paths") or [])],
    }


def _output_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    post = state.get("post") or {}
    slides = state.get("slides") or []
    critic = state.get("critic_report") or {}
    chunks = state.get("research_chunks") or []
    plan = state.get("plan") or {}

    return {
        "hook": post.get("hook"),
        "body": post.get("body"),
        "hashtags": post.get("hashtags"),
        "cta": post.get("cta"),
        "source_markers": post.get("source_markers"),
        "per_slide_captions": post.get("per_slide_captions"),
        "per_slide_bullets": post.get("per_slide_bullets"),
        "slide_count": len(slides),
        "slides": [
            {
                "index": s.get("index"),
                "title": s.get("title"),
                "treatment": s.get("treatment"),
                "caption": s.get("caption"),
                "rendered_path": s.get("rendered_path"),
            }
            for s in slides
        ],
        "plan_slide_titles": [sl.get("title") for sl in (plan.get("slides") or [])],
        "pdf_queries": plan.get("pdf_queries"),
        "research_chunk_count": len(chunks),
        "research_chunk_ids": [c.get("chunk_id") for c in chunks[:40]],
        "research_modalities": _modality_counts(chunks),
        "critic_pass": critic.get("pass"),
        "critic_iterations": state.get("critic_iterations"),
        "critic_route": critic.get("route"),
        "critic_scores": critic.get("scores"),
        "critic_issues": critic.get("issues"),
        "token_usage": state.get("token_usage"),
        "sources_count": len(state.get("sources") or []),
    }


def _modality_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        m = (c.get("metadata") or {}).get("modality", "unknown")
        counts[m] = counts.get(m, 0) + 1
    return counts


def log_interaction(
    run_id: str,
    initial: dict[str, Any],
    final: dict[str, Any],
    *,
    trace_path: str | None = None,
    source: str = "app",
    case_id: str | None = None,
    error: str | None = None,
    duration_ms: float | None = None,
) -> Path | None:
    """
    Append one JSON line to interactions.jsonl and write storage/interactions/<run_id>.json.

    Returns path to the per-run JSON file, or None if logging is disabled.
    """
    if not _enabled():
        return None

    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source": source,
        "case_id": case_id,
        "duration_ms": duration_ms,
        "error": error,
        "input": _input_snapshot(initial),
        "output": _output_snapshot(final) if not error else {},
        "trace_path": trace_path,
    }

    detail_path = interactions_dir() / f"{run_id}.json"
    detail_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")

    with _jsonl_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")

    print(f"[INTERACTION] logged run_id={run_id} → {_jsonl_path().name}", flush=True)
    return detail_path


def load_interactions(limit: int | None = None) -> list[dict[str, Any]]:
    """Load all interaction rows from the append-only JSONL log."""
    path = _jsonl_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if limit is not None:
        return rows[-limit:]
    return rows
