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
        "needs_web (bool), "
        "pdf_queries (array of 3-5 short search queries to run against the uploaded PDFs — "
        "infer what kinds of content the document likely contains based on the topic, "
        "e.g. for a resume: experience, projects, skills; for a research paper: methodology, results, contributions; "
        "for a product spec: features, use-cases, metrics — always derive from context, never hardcode), "
        "slides (array of {title, bullets: string[]}). "
        f"Target num_slides={state.get('num_slides', 3)}. Respect tone={state.get('tone')}.\n"
        f"Brand skill (excerpt):\n{brand[:3000]}"
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
    plan = state.get("plan") or {}
    # Use planner-generated queries; fall back to topic + broad summary if absent
    pdf_queries: list[str] = plan.get("pdf_queries") or []
    if not pdf_queries:
        pdf_queries = [topic] if topic else ["summary overview"]
    for pid in state.get("pdf_ids") or []:
        seen_chunk_ids: set[str] = set()
        for q in pdf_queries:
            res = tr.pdf_query(pid, q, k=6)
            trace.log("tool_call", tool="pdf_query", input={"pdf_id": pid, "query": q},
                      chunk_ids=[c["chunk_id"] for c in res.get("chunks", [])])
            for c in res.get("chunks", []):
                if c["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(c["chunk_id"])
                    chunks.append(c)
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
        f"[{c.get('chunk_id')}] {c.get('text','')[:1500]}" for c in state.get("research_chunks", [])[:24]
    )
    sys = (
        "You are the Copywriter. Your job is to write a LinkedIn post grounded ENTIRELY in the "
        "provided research_excerpt. RULES:\n"
        "1. Use ONLY facts, names, projects, technologies, metrics, and achievements that appear verbatim "
        "or are directly inferable from the research_excerpt. Do NOT invent, generalise, or use placeholder text.\n"
        "2. If the excerpt is a resume, extract the person's actual job titles, specific projects, "
        "real technologies used, and concrete results/numbers mentioned.\n"
        "3. Never write generic sentences like 'I worked on diverse projects' — always name the actual project.\n"
        "4. Populate source_markers with the chunk_ids you directly used.\n"
        "Return JSON keys: hook, body, hashtags (array of 3-5 relevant tags), cta, "
        "source_markers (array of chunk_ids you relied on), per_slide_captions (array aligned to plan slides).\n"
        f"LinkedIn skill:\n{li[:2000]}\nCitation skill:\n{cit[:1500]}\nBrand:\n{brand[:1500]}"
    )
    user = json.dumps(
        {
            "topic": state.get("topic"),
            "tone": state.get("tone"),
            "target_length": state.get("target_length", "medium"),
            "plan": state.get("plan", {}),
            "research_excerpt": ctx[:24000],
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
        caps = (state.get("post") or {}).get("per_slide_captions") or []
        cap = caps[i] if i < len(caps) else ""
        out_path = _render_mock_slide(
            i,
            title=sl.get("title", f"Slide {i+1}"),
            bullets=sl.get("bullets", []),
            caption=cap,
            prompt=prompt,
            color=state.get("brand_color") or "#1d3557",
        )
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


def _render_mock_slide(
    idx: int,
    title: str,
    bullets: list[str],
    caption: str,
    prompt: str,
    color: str,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1080, 1080
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "storage" / "extracted_images" / "mock_slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"mock_slide_{idx}.png"

    # Background
    img = Image.new("RGB", (W, H), color=color)
    d = ImageDraw.Draw(img)

    # Attempt to load a slightly larger default font
    try:
        font_title = ImageFont.load_default(size=48)
        font_body = ImageFont.load_default(size=28)
        font_small = ImageFont.load_default(size=20)
    except TypeError:
        # Older Pillow: load_default() takes no args
        font_title = font_body = font_small = ImageFont.load_default()

    # Slide number badge (top-left)
    badge = f"  {idx + 1}  "
    d.rectangle([40, 40, 120, 90], fill="white")
    d.text((50, 48), badge, fill=color, font=font_body)

    # MOCK watermark (top-right, semi-transparent feel via lighter colour)
    d.text((W - 160, 48), "MOCK", fill="#ffffff88" if hasattr(d, "fontmode") else "white", font=font_body)

    # Title
    y = 130
    _draw_wrapped(d, title.upper(), font_title, "white", 60, y, W - 60, line_height=58)
    y += _text_height(title, W - 120, 58) + 30

    # Divider line
    d.rectangle([60, y, W - 60, y + 4], fill="white")
    y += 24

    # Bullets
    for b in bullets[:6]:
        bullet_text = f"• {b}"
        _draw_wrapped(d, bullet_text, font_body, "#e0e0e0", 60, y, W - 60, line_height=36)
        y += _text_height(bullet_text, W - 120, 36) + 10

    # Caption (bottom)
    if caption:
        cap_y = H - 130
        d.rectangle([0, cap_y - 10, W, H - 60], fill="#00000055" if len(color) < 9 else color)
        _draw_wrapped(d, caption[:200], font_small, "#dddddd", 60, cap_y, W - 60, line_height=26)

    # Image prompt hint (very bottom, small)
    d.text((60, H - 52), f"Prompt: {prompt[:100]}", fill="#aaaaaa", font=font_small)

    img.save(path)
    return path


def _draw_wrapped(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.ImageFont",
                  fill: str, x: int, y: int, max_x: int, line_height: int) -> None:
    words = text.split()
    line: list[str] = []
    for word in words:
        test = " ".join(line + [word])
        try:
            w = draw.textlength(test, font=font)
        except Exception:
            w = len(test) * 9
        if w > (max_x - x) and line:
            draw.text((x, y), " ".join(line), fill=fill, font=font)
            y += line_height
            line = [word]
        else:
            line.append(word)
    if line:
        draw.text((x, y), " ".join(line), fill=fill, font=font)


def _text_height(text: str, max_width_chars: int, line_height: int) -> int:
    words = text.split()
    lines = max(1, len(words) // max(1, max_width_chars // 8))
    return lines * line_height


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
