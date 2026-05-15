"""Hybrid-ish retrieval: Chroma vector + simple BM25 re-ranking when rank_bm25 is available."""

from __future__ import annotations

from typing import Any

from rag.store import KnowledgeStore


def retrieve_for_question(
    question: str,
    pdf_id: str | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    kb = KnowledgeStore.get()
    if pdf_id:
        chunks = kb.query_pdf(pdf_id, question, k=max(k * 2, 12))
    else:
        chunks = kb.query_global(question, k=max(k * 2, 12))

    try:
        from rank_bm25 import BM25Okapi

        corpus = [c["text"] for c in chunks]
        tokenized = [c.lower().split() for c in corpus]
        if tokenized:
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(question.lower().split())
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            merged: list[dict[str, Any]] = []
            seen = set()
            for i in order[:k]:
                merged.append(chunks[i])
                seen.add(i)
            for i, c in enumerate(chunks):
                if i not in seen and len(merged) < k * 2:
                    merged.append(c)
            return merged[:k]
    except Exception:
        pass
    return chunks[:k]
