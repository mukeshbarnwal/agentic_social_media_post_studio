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

from observability.langsmith_setup import configure_langsmith, is_langsmith_enabled

configure_langsmith()

import streamlit as st

from graph.state import StudioState
from graph.workflow import run_studio
from mcp_server import tool_runtime as tr
from rag.store import project_root


def _upload_dir() -> Path:
    d = project_root() / os.getenv("UPLOAD_DIR", "storage/uploads")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inject_ui_styles() -> None:
    """Large base typography for readability on wide layouts."""
    st.markdown(
        """
        <style>
        html, body, .stApp {
            font-size: 20px !important;
        }
        .stApp h1 {
            font-size: 2.5rem !important;
            line-height: 1.2 !important;
        }
        .stApp h2 { font-size: 1.85rem !important; }
        .stApp h3, .stApp h4, .stApp h5 {
            font-size: 1.5rem !important;
        }
        .stApp p, .stApp li, .stApp label,
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li {
            font-size: 1.25rem !important;
            line-height: 1.55 !important;
        }
        .stApp textarea,
        .stApp input,
        .stApp [data-baseweb="input"] input,
        .stApp [data-baseweb="textarea"] textarea,
        .stApp [data-baseweb="select"] > div {
            font-size: 1.2rem !important;
            line-height: 1.5 !important;
        }
        .stApp button, .stApp button p {
            font-size: 1.2rem !important;
        }
        .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] {
            font-size: 1.1rem !important;
        }
        .stApp [data-testid="stAlert"] p,
        .stApp [data-testid="stAlert"] div {
            font-size: 1.15rem !important;
        }
        .stApp [data-testid="stDataFrame"] {
            font-size: 1.1rem !important;
        }
        .stApp pre, .stApp code {
            font-size: 1.05rem !important;
        }
        .stApp [data-testid="stSidebar"] {
            font-size: 1.15rem !important;
        }
        .stApp [data-testid="stSidebar"] label {
            font-size: 1.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Agentic Social Media Post Studio", layout="wide")
    _inject_ui_styles()
    st.title("Agentic Social Media Post Studio")
    st.caption("Multi-agent LinkedIn studio with MCP tools and Observability.")

    mock = os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes")

    with st.sidebar:
        st.header("Style")
        tone = st.selectbox("Tone", ["formal", "casual", "punchy"], index=1)
        target_length = st.selectbox("Target length", ["short", "medium", "long"], index=1)
        num_slides = st.slider("Slides (carousel)", 1, 8, 3)
        st.caption("Auto-set to 1 when only one image is uploaded")
        brand_color = st.color_picker("Brand tint (mock slides)", "#1d3557")

        st.divider()
        st.header("Observability")
        if mock:
            st.info("MOCK_MODELS on — stub LLM/search; full pipeline still runs.")
        if is_langsmith_enabled():
            project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "default"
            #st.success(f"LangSmith **{project}**")
            st.link_button("View LangSmith", "https://smith.langchain.com")
        else:
            st.caption("LangSmith is off — set `LANGSMITH_TRACING` + `LANGSMITH_API_KEY` in `.env`.")

    col_main, col_rerun = st.columns((3, 1))
    with col_main:
        topic = st.text_area("Topic / brief", height=160, placeholder="What should the post accomplish?")
        url_or_query = st.text_input("URL or web search query (optional)", placeholder="https://… or a short query")
        pdfs = st.file_uploader("PDFs (optional)", type=["pdf"], accept_multiple_files=True)
        imgs = st.file_uploader("Images (optional)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    with col_rerun:
        st.markdown("##### Partial rerun")
        #st.caption("Edit hook/body → copywriter → visual → critic (keeps plan + research).")
        edited_hook = st.text_input("Hook edit", placeholder="e.g. make it punchier")
        edited_body = st.text_area("Body edit", height=100, placeholder="e.g. emphasise impact")

    run = st.button("Generate post", type="primary")

    if "last_manifest" not in st.session_state:
        st.session_state.last_manifest = None
        st.session_state.last_state = None
        st.session_state.last_trace_path = None

    if run:
        up = _upload_dir()
        pdf_ids: list[str] = []

        step_placeholder = st.empty()

        def _show_step(msg: str) -> None:
            step_placeholder.caption(f"⏳ {msg}")

        if pdfs:
            for f in pdfs:
                dest = up / f.name
                dest.write_bytes(f.getvalue())
                print(f"[UI]     PDF saved to disk: {dest.name} ({len(f.getvalue())//1024} KB)", flush=True)
                _show_step(f"📄 Indexing **{f.name}** into Chroma…")
                res = tr.index_pdf(str(dest))
                if res.get("error"):
                    st.error(res["error"])
                    print(f"[UI]     index_pdf error: {res['error']}", flush=True)
                else:
                    pdf_ids.append(res["pdf_id"])
                    print(f"[UI]     Indexed → pdf_id={res['pdf_id']}", flush=True)

        image_paths: list[str] = []
        if imgs:
            for im in imgs:
                dest = up / im.name
                dest.write_bytes(im.getvalue())
                image_paths.append(str(dest))
                _show_step(f"🖼️ Uploaded image **{im.name}**")

        # Flow 3: image-only (no PDF, no URL) → single-image post
        is_image_only = bool(image_paths) and not pdf_ids and not url_or_query.strip()
        effective_slides = 1 if is_image_only else int(num_slides)
        if is_image_only:
            _show_step("🖼️ Image-only input detected — generating single-image post")

        initial: StudioState = {
            "topic": topic or "General LinkedIn update",
            "tone": tone,
            "target_length": target_length,  # type: ignore[assignment]
            "num_slides": effective_slides,
            "brand_color": brand_color,
            "pdf_ids": pdf_ids,
            "image_paths": image_paths,
            "url_or_query": url_or_query,
            "user_edited_hook": (edited_hook or "").strip() or None,
            "user_edited_body": (edited_body or "").strip() or None,
            "rerun_scope": "full",
            "critic_iterations": 0,
        }

        out, trace = run_studio(initial, status_callback=_show_step, log_source="streamlit")
        step_placeholder.empty()

        st.session_state.last_state = out
        st.session_state.last_manifest = out.get("manifest")
        st.session_state.last_trace_path = str(trace.path) if trace.path else None

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
        # Only pass fields the user actually edited (do not fall back to prior hook/body —
        # that tells the copywriter to keep them unchanged).
        slim["user_edited_hook"] = (edited_hook or "").strip() or None
        slim["user_edited_body"] = (edited_body or "").strip() or None
        if not slim["user_edited_hook"] and not slim["user_edited_body"]:
            st.warning("Enter a hook and/or body edit before partial rerun.")
            st.stop()
        slim["critic_iterations"] = 0
        slim["critic_report"] = {}
        p_placeholder = st.empty()

        def _show_partial(msg: str) -> None:
            p_placeholder.caption(f"⏳ {msg}")

        out, trace = run_studio(slim, status_callback=_show_partial, log_source="streamlit_partial")
        p_placeholder.empty()
        st.session_state.last_state = out
        st.session_state.last_manifest = out.get("manifest")
        st.session_state.last_trace_path = str(trace.path) if trace.path else None

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
                treatment = s.get("treatment", "")
                st.write(f"**Slide {s.get('index',0)+1} — {s.get('title','')}** ({treatment})")
                p = Path(s.get("rendered_path") or "")
                if p.exists():
                    st.image(str(p), caption=s.get("alt_text", ""))
                # Always show bullets for pdf_figure and uploaded_image treatments
                # (mock slides have bullets baked into the PNG; real assets don't)
                if treatment in ("pdf_figure", "uploaded_image"):
                    bullets = s.get("bullets") or []
                    if bullets:
                        for b in bullets:
                            st.markdown(f"- {b}")
                if s.get("caption"):
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
