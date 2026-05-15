"""Hybrid-ish retrieval: Chroma vector + simple BM25 re-ranking when rank_bm25 is available."""

from __future__ import annotations

from typing import Any

from rag.store import KnowledgeStore


def retrieve_for_question(
    question: str,
    pdf_id: str | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    scope = f"pdf_id={pdf_id}" if pdf_id else "global"
    print(f"[RETRIEVE] query={question[:80]!r} | scope={scope} | k={k}", flush=True)
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
            result = merged[:k]
            print(f"[RETRIEVE] BM25 re-rank → returning {len(result)} chunks", flush=True)
            return result
    except Exception:
        pass
    print(f"[RETRIEVE] Vector-only → returning {len(chunks[:k])} chunks", flush=True)
    return chunks[:k]
