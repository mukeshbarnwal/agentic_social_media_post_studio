"""LangGraph node functions (agents). MCP-equivalent tools: `mcp_server.tool_runtime`."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from graph.llm import chat_json, mock_models
from graph.state import StudioState
from mcp_server import tool_runtime as tr
from observability.trace_logger import RunTrace
from skill_loader import skill_prompt


def planner_node(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    if state.get("rerun_scope") == "copywriter":
        trace.log("agent_skip", agent="planner", reason="partial_rerun_copywriter")
        return {}
    usage: dict[str, int] = {}
    brand = skill_prompt("brand_voice")
    sys = (
        "You are the Planner/Orchestrator. Output JSON only with keys: "
        "needs_web (bool), slides (array of {title, bullets: string[]}). "
        f"Target num_slides={state.get('num_slides', 3)}. Respect tone={state.get('tone')}.\n"
        f"Brand skill (excerpt):\n{brand[:3500]}"
    )
    src = tr.list_sources()
    user = json.dumps(
        {
            "topic": state.get("topic"),
            "tone": state.get("tone"),
            "num_slides": state.get("num_slides", 3),
            "pdf_ids": state.get("pdf_ids", []),
            "url_or_query": state.get("url_or_query", ""),
            "sources_overview": {"distinct_pdf_ids": src.get("distinct_pdf_ids", []), "chunk_count": len(src.get("chunks", []))},
        },
        default=str,
    )
    plan = chat_json(sys, user, usage)
    trace.log(
        "agent_end",
        agent="planner",
        output_keys=list(plan.keys()),
        tool="list_sources",
        tool_output_summary={"pdfs": (src.get("distinct_pdf_ids") or [])[:5]},
    )
    return {"plan": plan, "token_usage": usage, "trace": [{"event": "planner", "plan": plan}]}


def research_node(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    if state.get("rerun_scope") == "copywriter":
        trace.log("agent_skip", agent="research", reason="partial_rerun_copywriter")
        return {}
    usage: dict[str, int] = {}
    chunks: list[dict[str, Any]] = []
    topic = state.get("topic", "")
    for pid in state.get("pdf_ids") or []:
        res = tr.pdf_query(pid, topic or "summary", k=8)
        trace.log("tool_call", tool="pdf_query", input={"pdf_id": pid}, chunk_ids=[c["chunk_id"] for c in res.get("chunks", [])])
        chunks.extend(res.get("chunks", []))
    uq = (state.get("url_or_query") or "").strip()
    if uq:
        if uq.startswith("http://") or uq.startswith("https://"):
            fr = tr.fetch_url(uq)
            trace.log("tool_call", tool="fetch_url", input={"url": uq}, output_keys=list(fr.keys()))
            if fr.get("markdown"):
                chunks.append(
                    {
                        "chunk_id": fr.get("web_source_id", "web:manual"),
                        "text": fr["markdown"][:6000],
                        "metadata": {"source_id": fr.get("web_source_id"), "modality": "web", "path": uq},
                    }
                )
        else:
            ws = tr.web_search(uq)
            trace.log("tool_call", tool="web_search", input={"query": uq})
            for r in ws.get("results", [])[:5]:
                sid = hashlib.md5((r.get("url") or "").encode()).hexdigest()[:10]
                chunks.append(
                    {
                        "chunk_id": f"websearch:{sid}",
                        "text": f"{r.get('title')}\n{r.get('snippet')}",
                        "metadata": {"source_id": f"websearch:{sid}", "modality": "web", "path": r.get("url")},
                    }
                )
    for img in state.get("image_paths") or []:
        cap = tr.caption_uploaded_image(img)
        trace.log("tool_call", tool="caption_uploaded_image", input={"path": img}, caption=cap.get("caption"))
        chunks.append(
            {
                "chunk_id": cap.get("image_id", img),
                "text": cap.get("caption", ""),
                "metadata": {"source_id": cap.get("image_id", img), "modality": "image", "path": img},
            }
        )
    return {
        "research_chunks": chunks,
        "token_usage": usage,
        "trace": [{"event": "research", "chunk_count": len(chunks)}],
    }


def copywriter_node(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    usage: dict[str, int] = {}
    li = skill_prompt("linkedin_formatting")
    cit = skill_prompt("citation")
    brand = skill_prompt("brand_voice")
    ctx = "\n\n".join(
        f"[{c.get('chunk_id')}] {c.get('text','')[:1200]}" for c in state.get("research_chunks", [])[:24]
    )
    sys = (
        "You are the Copywriter. Return JSON keys: hook, body, hashtags (array), cta, "
        "source_markers (array of chunk_ids you relied on), per_slide_captions (array aligned to plan slides).\n"
        f"LinkedIn skill:\n{li[:2500]}\nCitation skill:\n{cit[:2500]}\nBrand:\n{brand[:2000]}"
    )
    user = json.dumps(
        {
            "topic": state.get("topic"),
            "tone": state.get("tone"),
            "target_length": state.get("target_length", "medium"),
            "plan": state.get("plan", {}),
            "research_excerpt": ctx[:20000],
            "user_edited_hook": state.get("user_edited_hook"),
            "user_edited_body": state.get("user_edited_body"),
        },
        default=str,
    )
    post = chat_json(sys, user, usage)
    if state.get("user_edited_hook"):
        post["hook"] = state["user_edited_hook"]
    if state.get("user_edited_body"):
        post["body"] = state["user_edited_body"]
    trace.log("agent_end", agent="copywriter", source_markers=post.get("source_markers", []))
    return {"post": post, "token_usage": usage, "trace": [{"event": "copywriter", "post": post}]}


def visual_node(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    """Decision rule: (1) uploaded images win, (2) else PDF figure chunk path, (3) else MOCK generated slide."""
    usage: dict[str, int] = {}
    ip = skill_prompt("image_prompting")
    slides_out: list[dict[str, Any]] = []
    plan_slides = (state.get("plan") or {}).get("slides") or [{"title": "Slide 1", "bullets": []}]
    imgs = state.get("image_paths") or []
    fig_paths = [
        c["metadata"].get("path")
        for c in state.get("research_chunks", [])
        if c.get("metadata", {}).get("modality") == "figure" and c.get("metadata", {}).get("path")
    ]
    for i, sl in enumerate(plan_slides):
        treatment = "generate_mock"
        asset_path = ""
        if imgs:
            treatment = "uploaded_image"
            asset_path = imgs[0]
        elif fig_paths:
            treatment = "pdf_figure"
            asset_path = fig_paths[min(i, len(fig_paths) - 1)]
        sys = "You are the Visual agent. Return JSON: image_prompt (string), alt_text (string), treatment echo."
        user = json.dumps({"slide": sl, "treatment": treatment, "asset_path": asset_path, "skill": ip[:2000]}, default=str)
        vis = chat_json(sys, user, usage) if not mock_models() else {}
        alt = vis.get("alt_text") or f"Alt text for slide {i+1}: {sl.get('title','')}"
        prompt = vis.get("image_prompt") or f"LinkedIn illustration for: {sl.get('title')}"
        out_path = _render_mock_slide(i, prompt, state.get("brand_color") or "#1d3557")
        caps = (state.get("post") or {}).get("per_slide_captions") or []
        cap = caps[i] if i < len(caps) else ""
        slides_out.append(
            {
                "index": i,
                "title": sl.get("title"),
                "bullets": sl.get("bullets", []),
                "caption": cap,
                "treatment": treatment,
                "asset_path": asset_path or str(out_path),
                "rendered_path": str(out_path),
                "image_prompt": prompt,
                "alt_text": alt,
            }
        )
    trace.log("agent_end", agent="visual", slides=len(slides_out))
    return {"slides": slides_out, "token_usage": usage, "trace": [{"event": "visual", "slides": slides_out}]}


def _render_mock_slide(idx: int, prompt: str, color: str) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    root = Path(__file__).resolve().parents[1]
    out_dir = root / "storage" / "extracted_images" / "mock_slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mock_slide_{idx}.png"
    img = Image.new("RGB", (1080, 1080), color=color)
    d = ImageDraw.Draw(img)
    text = f"MOCK slide {idx+1}\n{prompt[:180]}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.multiline_text((40, 40), text, fill="white", font=font, spacing=4)
    img.save(path)
    return path


def critic_node(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    usage: dict[str, int] = {}
    rub = skill_prompt("critic_rubric")
    post = state.get("post") or {}
    markers = set(post.get("source_markers") or [])
    chunk_ids = {c.get("chunk_id") for c in state.get("research_chunks", [])}
    grounded = bool(markers & chunk_ids) or mock_models()
    hashtags = post.get("hashtags") or []
    fmt_ok = isinstance(hashtags, list) and 1 <= len(hashtags) <= 6
    sys = (
        "You are the Critic/QA. Return JSON: pass (bool), scores {grounding, format, voice}, "
        "issues[], route one of end|copywriter|visual|research.\n"
        f"Rubric:\n{rub[:4000]}"
    )
    user = json.dumps(
        {
            "post": post,
            "grounding_precheck": grounded,
            "format_precheck": fmt_ok,
            "slides": state.get("slides", []),
        },
        default=str,
    )
    report = chat_json(sys, user, usage)
    if not report:
        report = {
            "pass": grounded and fmt_ok,
            "scores": {"grounding": 0.9 if grounded else 0.4, "format": 0.9 if fmt_ok else 0.5, "voice": 0.85},
            "issues": [] if grounded and fmt_ok else ["Add explicit source_markers from research chunk ids."],
            "route": "end" if grounded and fmt_ok else "copywriter",
        }
    trace.log("agent_end", agent="critic", report=report)
    it = int(state.get("critic_iterations") or 0) + 1
    return {"critic_report": report, "critic_iterations": it, "token_usage": usage, "trace": [{"event": "critic", "report": report}]}


def assemble_manifest(state: StudioState, trace: RunTrace) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in state.get("research_chunks", []):
        sid = c.get("chunk_id")
        if sid in seen:
            continue
        seen.add(sid)
        md = c.get("metadata") or {}
        sources.append(
            {
                "source_id": md.get("source_id", sid),
                "chunk_id": sid,
                "modality": md.get("modality"),
                "page": md.get("page"),
                "path": md.get("path"),
            }
        )
    manifest = {
        "topic": state.get("topic"),
        "post": state.get("post"),
        "slides": state.get("slides"),
        "sources": sources,
        "plan": state.get("plan"),
        "critic_report": state.get("critic_report"),
        "token_usage": state.get("token_usage", {}),
    }
    trace.log("run_complete", manifest_keys=list(manifest.keys()))
    return {"manifest": manifest, "sources": sources, "trace": [{"event": "assemble"}]}
