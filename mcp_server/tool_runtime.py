"""Sync implementations backing MCP tools (also callable in-process for Streamlit)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from rag.ingest_pdf import ingest_pdf_to_store
from rag.retriever import retrieve_for_question
from rag.store import KnowledgeStore, project_root


def _uploads_dir() -> Path:
    return project_root() / os.getenv("UPLOAD_DIR", "storage/uploads")


def _extracted_dir() -> Path:
    return project_root() / os.getenv("EXTRACTED_IMAGES_DIR", "storage/extracted_images")


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Ranked web results. MOCK unless TAVILY_API_KEY is set."""
    if os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes") and not os.getenv("TAVILY_API_KEY"):
        h = hashlib.md5(query.encode()).hexdigest()[:8]
        results = [
            {
                "title": f"MOCK result A ({h})",
                "url": f"https://example.invalid/mock-a/{h}",
                "snippet": f"Deterministic MOCK snippet for query: {query[:120]}",
            },
            {
                "title": f"MOCK result B ({h})",
                "url": f"https://example.invalid/mock-b/{h}",
                "snippet": "MOCK_MODELS=true: replace with Tavily or live search by setting keys.",
            },
        ]
        return {"query": query, "results": results[:max_results], "mock": True}

    key = os.getenv("TAVILY_API_KEY")
    if key:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": max_results},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        results = [{"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content")} for x in data.get("results", [])]
        return {"query": query, "results": results, "mock": False}

    # Free fallback: DuckDuckGo HTML (best effort)
    try:
        q = httpx.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 StudioBot/1.0"},
            timeout=20.0,
        )
        q.raise_for_status()
        soup = BeautifulSoup(q.text, "lxml")
        rows: list[dict[str, Any]] = []
        for tr in soup.select("table:nth-of-type(2) tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            a = tds[0].find("a")
            if not a or not a.get("href"):
                continue
            title = a.get_text(strip=True)
            href = a["href"]
            snippet = tds[1].get_text(" ", strip=True)
            if title and href.startswith("http"):
                rows.append({"title": title, "url": href, "snippet": snippet})
            if len(rows) >= max_results:
                break
        return {"query": query, "results": rows, "mock": False}
    except Exception as exc:
        return {"query": query, "results": [], "error": str(exc), "mock": False}


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch URL, strip boilerplate, return markdown-ish text + image URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"url": url, "error": "Only http(s) URLs are allowed."}
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 StudioBot/1.0"}, timeout=25.0, follow_redirects=True)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        texts: list[str] = []
        for el in soup.find_all(["article", "main", "p", "li", "h1", "h2", "h3"]):
            t = el.get_text(" ", strip=True)
            if len(t) > 40:
                texts.append(t)
        markdown = "\n\n".join(texts[:200])[:120_000]
        imgs: list[str] = []
        for im in soup.find_all("img", src=True)[:30]:
            src = im["src"].strip()
            if src.startswith("//"):
                src = f"{parsed.scheme}:{src}"
            elif src.startswith("/"):
                src = f"{parsed.scheme}://{parsed.netloc}{src}"
            if src.startswith("http"):
                imgs.append(src)
        wid = KnowledgeStore.get().register_web_page(url, markdown or soup.get_text("\n", strip=True)[:80_000], imgs)
        return {"url": url, "markdown": markdown, "image_urls": imgs, "web_source_id": wid}
    except Exception as exc:
        return {"url": url, "error": str(exc)}


def pdf_query(pdf_id: str, question: str, k: int = 6) -> dict[str, Any]:
    chunks = retrieve_for_question(question, pdf_id=pdf_id, k=k)
    return {
        "pdf_id": pdf_id,
        "question": question,
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"][:4000],
                "metadata": c["metadata"],
            }
            for c in chunks
        ],
    }


def index_pdf(file_path: str) -> dict[str, Any]:
    raw = Path(file_path).expanduser()
    uploads = _uploads_dir().resolve()
    path = (uploads / raw.name).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        path.relative_to(uploads)
    except ValueError:
        return {"error": f"PDF must be under uploads directory: {uploads}"}
    if not path.exists():
        return {"error": f"File not found: {path}"}
    pdf_id = ingest_pdf_to_store(path, _extracted_dir())
    return {"pdf_id": pdf_id, "path": str(path)}


def list_sources() -> dict[str, Any]:
    rows = KnowledgeStore.get().list_sources()
    pdf_ids = sorted({r.get("pdf_id") for r in rows if r.get("pdf_id")})
    return {"chunks": rows, "distinct_pdf_ids": [p for p in pdf_ids if p]}


def caption_uploaded_image(file_path: str) -> dict[str, Any]:
    """MOCK vision caption unless OPENAI_API_KEY and MOCK_MODELS is false."""
    path = Path(file_path).resolve()
    if os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes") or not os.getenv("OPENAI_API_KEY"):
        cap = f"MOCK caption for {path.name}: user-supplied image suitable for a LinkedIn visual."
        iid = KnowledgeStore.get().register_image_caption(str(path), cap)
        return {"image_id": iid, "caption": cap, "mock": True}
    try:
        from openai import OpenAI

        client = OpenAI()
        mime, b64 = _b64(path)
        r = client.chat.completions.create(
                model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Write a concise neutral image description for accessibility and marketing."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        ],
                    }
                ],
                max_tokens=200,
            )
        cap = r.choices[0].message.content or ""
        iid = KnowledgeStore.get().register_image_caption(str(path), cap)
        return {"image_id": iid, "caption": cap, "mock": False}
    except Exception as exc:
        cap = f"MOCK caption (vision failed: {exc})"
        iid = KnowledgeStore.get().register_image_caption(str(path), cap)
        return {"image_id": iid, "caption": cap, "mock": True}


def _b64(path: Path) -> tuple[str, str]:
    import base64
    import mimetypes

    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    b = base64.standard_b64encode(path.read_bytes()).decode()
    return mime, b
