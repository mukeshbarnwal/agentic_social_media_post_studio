"""PDF ingestion: text per page, embedded figures to disk, table-ish blocks as text."""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from rag.chunking import chunk_text
from rag.store import KnowledgeStore


def extract_pdf_assets(
    pdf_path: Path,
    pdf_id: str,
    extracted_dir: Path,
) -> list[dict[str, Any]]:
    """Return list of chunk dicts: text, table, or figure rows with metadata."""
    doc = fitz.open(pdf_path)
    chunks_meta: list[dict[str, Any]] = []
    extracted_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        text = page.get_text("text") or ""
        for ci, chunk in enumerate(chunk_text(text)):
            sid = f"{pdf_id}:p{page_num}:t{ci}"
            chunks_meta.append(
                {
                    "chunk_id": sid,
                    "text": chunk,
                    "metadata": {
                        "source_id": sid,
                        "pdf_id": pdf_id,
                        "page": page_num,
                        "modality": "text",
                        "path": str(pdf_path),
                    },
                }
            )

        # Tables: best-effort text blocks that look tabular
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        ti = 0
        for block in blocks:
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            joined = "\n".join(
                "".join(span.get("text", "") for span in line.get("spans", [])) for line in lines
            ).strip()
            if len(joined) < 8:
                continue
            if "\t" in joined or re.search(r"\s{3,}", joined) or re.search(r"\d+\s+\|\s+", joined):
                sid = f"{pdf_id}:p{page_num}:tbl{ti}"
                ti += 1
                chunks_meta.append(
                    {
                        "chunk_id": sid,
                        "text": f"[table-like block p{page_num}]\n{joined}",
                        "metadata": {
                            "source_id": sid,
                            "pdf_id": pdf_id,
                            "page": page_num,
                            "modality": "table",
                            "path": str(pdf_path),
                        },
                    }
                )

        # Embedded images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            sid = f"{pdf_id}:p{page_num}:fig{img_index}"
            try:
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base.get("ext", "png")
                out = extracted_dir / f"{sid.replace(':', '_')}.{ext}"
                out.write_bytes(img_bytes)
                cap = f"Figure extracted from PDF page {page_num} (image {img_index})."
                chunks_meta.append(
                    {
                        "chunk_id": sid,
                        "text": cap,
                        "metadata": {
                            "source_id": sid,
                            "pdf_id": pdf_id,
                            "page": page_num,
                            "modality": "figure",
                            "path": str(out),
                            "caption": cap,
                        },
                    }
                )
            except Exception:
                continue

    doc.close()
    return chunks_meta


def new_pdf_id() -> str:
    return f"pdf_{uuid.uuid4().hex[:12]}"


def ingest_pdf_to_store(pdf_path: Path, extracted_dir: Path) -> str:
    """Parse PDF, persist figure crops, upsert chunks into Chroma. Returns pdf_id."""
    pdf_id = new_pdf_id()
    items = extract_pdf_assets(pdf_path, pdf_id, extracted_dir)
    KnowledgeStore.get().add_chunks(items)
    return pdf_id
