"""Embedding providers: OpenAI (real) or deterministic MOCK for CI / laptops without keys."""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

from chromadb import Documents, EmbeddingFunction, Embeddings


class MockEmbeddingFunction(EmbeddingFunction):
    """MOCK: deterministic dense vectors from text (1536-dim).

    Dimension matches OpenAI text-embedding-3-small so a developer can switch
    from MOCK_MODELS=true to a real API key without a Chroma dimension mismatch.
    No external model download required.
    """

    DIM = 1536

    def __call__(self, input: Documents) -> Embeddings:
        out: Embeddings = []
        for text in input:
            h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
            vec: list[float] = []
            for i in range(self.DIM):
                vec.append((((h[i % len(h)] + i * 31) % 256) / 127.5) - 1.0)
            out.append(vec)
        return out


def mock_models() -> bool:
    return os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes")


def build_embedding_function() -> EmbeddingFunction:
    if mock_models():
        return MockEmbeddingFunction()
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return MockEmbeddingFunction()
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    return OpenAIEmbeddingFunction(api_key=key, model_name=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
