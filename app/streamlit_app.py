"""Streamlit UI: uploads, generation, sources panel, trace viewer, partial rerun."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure the project root (parent of this file's directory) is always on sys.path
# regardless of how streamlit resolves the working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st

from graph.state import StudioState
from graph.workflow import run_studio
from mcp_server import tool_runtime as tr
from rag.store import project_root


def _upload_dir() -> Path:
    d = project_root() / os.getenv("UPLOAD_DIR", "storage/uploads")
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> None:
    st.set_page_config(page_title="Agentic Social Post Studio", layout="wide")
    st.title("Agentic Social Post Studio")
    st.caption("Multi-agent LinkedIn studio with MCP tools, Chroma RAG, and MOCK_MODELS-friendly stubs.")

    mock = os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes")
    if mock:
        st.info("MOCK_MODELS is enabled: deterministic LLM + search stubs; full flow still runs.")

    with st.sidebar:
        st.header("Style")
        tone = st.selectbox("Tone", ["formal", "casual", "punchy"], index=1)
        target_length = st.selectbox("Target length", ["short", "medium", "long"], index=1)
        num_slides = st.slider("Slides (carousel)", 1, 8, 3)
        brand_color = st.color_picker("Brand tint (mock slides)", "#1d3557")

    col1, col2 = st.columns((1, 1))
    with col1:
        topic = st.text_area("Topic / brief", height=140, placeholder="What should the post accomplish?")
        url_or_query = st.text_input("URL or web search query (optional)", placeholder="https://… or a short query")
        pdfs = st.file_uploader("PDFs (optional)", type=["pdf"], accept_multiple_files=True)
        imgs = st.file_uploader("Images (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    with col2:
        st.subheader("Partial rerun (no full regeneration)")
        st.caption("Edit copy, then rerun copywriter → visual → critic while keeping prior research/plan.")
        edited_hook = st.text_input("Override hook (optional)")
        edited_body = st.text_area("Override body (optional)", height=120)

    run = st.button("Generate post", type="primary")

    if "last_manifest" not in st.session_state:
        st.session_state.last_manifest = None
        st.session_state.last_state = None
        st.session_state.last_trace_path = None

    if run:
        up = _upload_dir()
        pdf_ids: list[str] = []
        if pdfs:
            for f in pdfs:
                dest = up / f.name
                dest.write_bytes(f.getvalue())
                res = tr.index_pdf(str(dest))
                if res.get("error"):
                    st.error(res["error"])
                else:
                    pdf_ids.append(res["pdf_id"])
        image_paths: list[str] = []
        if imgs:
            for im in imgs:
                dest = up / im.name
                dest.write_bytes(im.getvalue())
                image_paths.append(str(dest))

        initial: StudioState = {
            "topic": topic or "General LinkedIn update",
            "tone": tone,
            "target_length": target_length,  # type: ignore[assignment]
            "num_slides": int(num_slides),
            "brand_color": brand_color,
            "pdf_ids": pdf_ids,
            "image_paths": image_paths,
            "url_or_query": url_or_query,
            "user_edited_hook": edited_hook or None,
            "user_edited_body": edited_body or None,
            "rerun_scope": "full",
            "critic_iterations": 0,
        }
        with st.spinner("Running planner → research → copywriter → visual → critic…"):
            out, trace = run_studio(initial)
        st.session_state.last_state = out
        st.session_state.last_manifest = out.get("manifest")
        st.session_state.last_trace_path = str(trace.path) if trace.path else None
        st.success("Run complete")

    partial = st.button("Rerun from edited copy only", help="Uses last run's plan + research chunks.")
    if partial and st.session_state.last_state:
        prev = dict(st.session_state.last_state)
        allowed = (
            "topic",
            "tone",
            "target_length",
            "num_slides",
            "brand_color",
            "pdf_ids",
            "image_paths",
            "url_or_query",
            "plan",
            "research_chunks",
            "post",
        )
        slim: StudioState = {k: prev[k] for k in allowed if k in prev}  # type: ignore[misc]
        slim["rerun_scope"] = "copywriter"
        slim["user_edited_hook"] = edited_hook or (prev.get("post") or {}).get("hook")
        slim["user_edited_body"] = edited_body or (prev.get("post") or {}).get("body")
        slim["critic_iterations"] = 0
        slim["critic_report"] = {}
        with st.spinner("Partial regeneration…"):
            out, trace = run_studio(slim)
        st.session_state.last_state = out
        st.session_state.last_manifest = out.get("manifest")
        st.session_state.last_trace_path = str(trace.path) if trace.path else None
        st.success("Partial rerun complete")

    man = st.session_state.last_manifest
    if man:
        st.divider()
        st.subheader("LinkedIn preview")
        post = man.get("post") or {}
        st.markdown(f"**{post.get('hook','')}**")
        st.write(post.get("body", ""))
        tags = post.get("hashtags") or []
        st.markdown(" ".join(f"`{t}`" for t in tags))
        st.caption(post.get("cta", ""))

        st.subheader("Carousel / visuals")
        for s in man.get("slides") or []:
            with st.container(border=True):
                st.write(f"**Slide {s.get('index',0)+1} — {s.get('title','')}** ({s.get('treatment')})")
                p = Path(s.get("rendered_path") or "")
                if p.exists():
                    st.image(str(p), caption=s.get("alt_text", ""))
                st.caption(s.get("caption", ""))

        st.subheader("Sources panel")
        st.dataframe(man.get("sources", []), use_container_width=True)

        st.subheader("Critic report")
        st.json(man.get("critic_report", {}))

        st.subheader("Token usage (aggregated)")
        st.json(man.get("token_usage", {}))

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download manifest JSON",
                data=json.dumps(man, indent=2, default=str),
                file_name="manifest.json",
                mime="application/json",
            )
        with c2:
            tp = st.session_state.last_trace_path
            if tp and Path(tp).exists():
                st.download_button(
                    "Download trace JSONL",
                    data=Path(tp).read_text(encoding="utf-8"),
                    file_name="trace.jsonl",
                    mime="text/plain",
                )

        st.subheader("Most recent trace (tail)")
        tp = st.session_state.last_trace_path
        if tp and Path(tp).exists():
            lines = Path(tp).read_text(encoding="utf-8").splitlines()[-40:]
            st.code("\n".join(lines), language="json")


if __name__ == "__main__":
    main()
