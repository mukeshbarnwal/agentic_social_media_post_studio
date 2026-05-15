"""Typed shared state for LangGraph (blackboard-style)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


def merge_token_dict(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0) + int(v)
    return out


class StudioState(TypedDict, total=False):
    topic: str
    tone: str
    target_length: Literal["short", "medium", "long"]
    num_slides: int
    brand_color: str | None
    pdf_ids: list[str]
    image_paths: list[str]
    url_or_query: str
    """User caption edits trigger partial regeneration."""
    user_edited_hook: str | None
    user_edited_body: str | None
    rerun_scope: Literal["full", "copywriter", "visual", "research"]
    plan: dict[str, Any]
    research_chunks: list[dict[str, Any]]
    post: dict[str, Any]
    slides: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    critic_report: dict[str, Any]
    critic_iterations: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    token_usage: Annotated[dict[str, int], merge_token_dict]
    manifest: dict[str, Any]
