"""Chroma-backed knowledge store with metadata for citations."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import chromadb

from rag.embeddings import build_embedding_function


def project_root() -> Path:
    return Path(os.getenv("STUDIO_ROOT", Path(__file__).resolve().parents[1]))


def chroma_path() -> Path:
    return project_root() / os.getenv("CHROMA_PERSIST_DIR", "storage/chroma")


class KnowledgeStore:
    """Singleton-style vector store + sidecar JSON for web/image sources not in Chroma."""

    _instance: KnowledgeStore | None = None

    def __init__(self) -> None:
        chroma_path().mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path()))
        self._collection = self._client.get_or_create_collection(
            name="studio_kb",
            embedding_function=build_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        self._extras_path = chroma_path() / "extras.json"
        self._extras: dict[str, Any] = {}
        if self._extras_path.exists():
            self._extras = json.loads(self._extras_path.read_text())

    @classmethod
    def get(cls) -> KnowledgeStore:
        if cls._instance is None:
            cls._instance = KnowledgeStore()
        return cls._instance

    def _persist_extras(self) -> None:
        self._extras_path.write_text(json.dumps(self._extras, indent=2))

    def add_chunks(self, items: list[dict[str, Any]]) -> None:
        if not items:
            return
        ids = [x["chunk_id"] for x in items]
        docs = [x["text"] for x in items]
        metas = [x["metadata"] for x in items]
        self._collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def query_pdf(self, pdf_id: str, question: str, k: int = 6) -> list[dict[str, Any]]:
        res = self._collection.query(
            query_texts=[question],
            n_results=k,
            where={"pdf_id": pdf_id},
        )
        out: list[dict[str, Any]] = []
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        for i in range(len(ids[0])):
            out.append(
                {
                    "chunk_id": ids[0][i],
                    "text": docs[0][i],
                    "metadata": metas[0][i],
                    "distance": dists[0][i] if dists and dists[0] else None,
                }
            )
        return out

    def query_global(self, question: str, k: int = 8) -> list[dict[str, Any]]:
        res = self._collection.query(query_texts=[question], n_results=k)
        out: list[dict[str, Any]] = []
        ids = res.get("ids") or [[]]
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        for i in range(len(ids[0])):
            out.append(
                {
                    "chunk_id": ids[0][i],
                    "text": docs[0][i],
                    "metadata": metas[0][i],
                    "distance": dists[0][i] if dists and dists[0] else None,
                }
            )
        return out

    def register_web_page(self, url: str, markdown: str, image_paths: list[str]) -> str:
        wid = f"web_{uuid.uuid4().hex[:10]}"
        chunks: list[dict[str, Any]] = []
        from rag.chunking import chunk_text

        for ci, chunk in enumerate(chunk_text(markdown, max_chars=1800, overlap=150)):
            sid = f"{wid}:c{ci}"
            chunks.append(
                {
                    "chunk_id": sid,
                    "text": chunk,
                    "metadata": {
                        "source_id": sid,
                        "pdf_id": "",
                        "page": 0,
                        "modality": "web",
                        "path": url,
                    },
                }
            )
        self.add_chunks(chunks)
        self._extras[wid] = {"url": url, "images": image_paths}
        self._persist_extras()
        return wid

    def register_image_caption(self, path: str, caption: str) -> str:
        iid = f"img_{uuid.uuid4().hex[:10]}"
        sid = f"{iid}:cap0"
        self.add_chunks(
            [
                {
                    "chunk_id": sid,
                    "text": caption,
                    "metadata": {
                        "source_id": sid,
                        "pdf_id": "",
                        "page": 0,
                        "modality": "image",
                        "path": path,
                    },
                }
            ]
        )
        return iid

    def list_sources(self) -> list[dict[str, Any]]:
        """Enumerate indexed chunks (capped) plus registered web bundles."""
        data = self._collection.get(include=["metadatas"], limit=800)
        rows: list[dict[str, Any]] = []
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        for i, cid in enumerate(ids):
            m = metas[i] if i < len(metas) else {}
            rows.append(
                {
                    "chunk_id": cid,
                    "source_id": m.get("source_id", cid),
                    "pdf_id": m.get("pdf_id", ""),
                    "page": m.get("page"),
                    "modality": m.get("modality"),
                    "path": m.get("path"),
                }
            )
        for wid, meta in self._extras.items():
            rows.append({"chunk_id": wid, "modality": "web_bundle", "path": meta.get("url")})
        return rows
