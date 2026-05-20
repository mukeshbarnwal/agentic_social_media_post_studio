"""LLM calls with MOCK_MODELS deterministic JSON fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def mock_models() -> bool:
    return os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes")


def _acc_tokens(usage: dict[str, int], inp: int, out: int) -> None:
    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + inp
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + out


def chat_json(system: str, user: str, usage: dict[str, int]) -> dict[str, Any]:
    """Return parsed JSON object from model."""
    if mock_models() or not os.getenv("OPENAI_API_KEY"):
        text = user + system
        _acc_tokens(usage, len(text) // 4, 80)
        return _mock_json_from_prompt(system, user)
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.2).bind(response_format={"type": "json_object"})
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    raw = msg.content
    if isinstance(raw, list):
        raw = "".join(str(x) for x in raw)
    data = json.loads(str(raw))
    meta = getattr(msg, "response_metadata", {}) or {}
    tok = meta.get("token_usage") or {}
    _acc_tokens(usage, int(tok.get("prompt_tokens", 0)), int(tok.get("completion_tokens", 0)))
    return data


def chat_text(system: str, user: str, usage: dict[str, int]) -> str:
    if mock_models() or not os.getenv("OPENAI_API_KEY"):
        text = user + system
        _acc_tokens(usage, len(text) // 4, 120)
        return f"MOCK narrative summary:\n{user[:800]}"
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model, temperature=0.3)
    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    meta = getattr(msg, "response_metadata", {}) or {}
    tok = meta.get("token_usage") or {}
    _acc_tokens(usage, int(tok.get("prompt_tokens", 0)), int(tok.get("completion_tokens", 0)))
    return str(msg.content)


def _mock_json_from_prompt(system: str, user: str) -> dict[str, Any]:
    s = system.lower()
    # Order matters: critic prompt embeds the word "copywriter" from the rubric.
    if "you are the planner" in s or "orchestrator" in s:
        m = re.search(r'"num_slides"\s*:\s*(\d+)', user)
        if not m:
            m = re.search(r"num_slides[^\d]*(\d+)", user)
        n = int(m.group(1)) if m else 3
        slides = [{"title": f"Slide {i + 1}", "bullets": [f"Key point {i + 1}a", f"Key point {i + 1}b"]} for i in range(n)]
        return {"needs_web": "http://" in user.lower() or "https://" in user.lower(), "slides": slides}
    if "you are the visual agent" in s:
        try:
            payload = json.loads(user)
            treatment = payload.get("treatment", "generate_mock")
        except Exception:
            treatment = "generate_mock"
        return {
            "image_prompt": f"MOCK minimalist illustration ({treatment})",
            "alt_text": "MOCK alt text describing the slide intent.",
            "treatment": treatment,
        }
    if "you are the critic" in s:
        return {
            "pass": True,
            "scores": {"grounding": 0.86, "format": 0.84, "voice": 0.83},
            "issues": [],
            "route": "end",
        }
    if "you are the copywriter" in s:
        ids = re.findall(r"\[([^\]]+)\]", user)
        markers = [i for i in ids if ":" in i][:8] or ["mock:chunk"]
        nslides = len(re.findall(r'"title"', user)) or 3
        captions = [f"MOCK caption for slide {j + 1}" for j in range(min(nslides, 8))]
        hook = "MOCK hook — grounded run (MOCK_MODELS=true)."
        body = "MOCK body paragraph one.\n\nMOCK body paragraph two with a concrete claim tied to sources."
        eh = re.search(r'"edit_hook"\s*:\s*"([^"]*)"', user)
        eb = re.search(r'"edit_body"\s*:\s*"([^"]*)"', user)
        if eh and eh.group(1).strip():
            hook = f"MOCK hook (edited): {eh.group(1).strip()[:120]}"
        if eb and eb.group(1).strip():
            body = f"MOCK body (edited): {eb.group(1).strip()[:500]}"
        return {
            "hook": hook,
            "body": body,
            "hashtags": ["#MockModel", "#LinkedIn", "#AI"],
            "cta": "MOCK CTA: comment if you want the real model path enabled.",
            "source_markers": markers[:5],
            "per_slide_captions": captions,
        }
    return {}
