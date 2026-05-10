"""
hybrid_retriever.py — Reciprocal Rank Fusion over BM25 + dense retrieval.

RRF combines ranked lists without needing score normalisation:
  RRF_score(d) = sum(1 / (k + rank_i(d)))  for each retriever i

This approach consistently outperforms individual retrievers on short,
ambiguous, or domain-specific queries — the common case in procurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Retriever interface
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return list of (product_id, score) sorted descending."""
        ...


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Keyword retrieval using rank_bm25.
    Tokenises product name + description.
    """

    def __init__(self, catalog: list[dict]):
        from rank_bm25 import BM25Okapi

        self.ids = [p["product_id"] for p in catalog]
        corpus = [
            self._tokenize(p["name"] + " " + p["description"])
            for p in catalog
        ]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_indices]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re
        return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Dense (bi-encoder) retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Semantic retrieval using sentence-transformers + FAISS index.

    Recommended models (configure in config.yaml):
      - all-MiniLM-L6-v2         (fast, general-purpose baseline)
      - medcpt-query-encoder      (domain-tuned on biomedical queries)
    """

    def __init__(self, catalog: list[dict], model_name: str = "all-MiniLM-L6-v2"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.ids = [p["product_id"] for p in catalog]
        self.model = SentenceTransformer(model_name)

        texts = [p["name"] + ". " + p["description"] for p in catalog]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=256)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine on normalised vecs
        self.index.add(embeddings.astype("float32"))

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = self.model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        scores, indices = self.index.search(q_emb.astype("float32"), top_k)
        return [(self.ids[i], float(s)) for i, s in zip(indices[0], scores[0])]


# ---------------------------------------------------------------------------
# Hybrid retriever (RRF)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Reciprocal Rank Fusion over an arbitrary number of retrievers.

    RRF_score(d) = Σ_i  1 / (k + rank_i(d))
    where k=60 is a standard smoothing constant (Cormack et al. 2009).
    """

    def __init__(self, retrievers: list[Retriever], k: int = 60):
        self.retrievers = retrievers
        self.k = k

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        rrf_scores: dict[str, float] = {}

        for retriever in self.retrievers:
            # fetch more candidates than top_k so RRF has enough to rerank
            candidates = retriever.retrieve(query, top_k=top_k * 3)
            for rank, (doc_id, _score) in enumerate(candidates, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retrievers(catalog: list[dict], config: dict) -> dict[str, Retriever]:
    """
    Instantiate all retrievers from config. Returns dict for benchmarking.
    """
    print("Building BM25 index...")
    bm25 = BM25Retriever(catalog)

    print(f"Building dense index ({config['dense_model']})...")
    dense = DenseRetriever(catalog, model_name=config["dense_model"])

    hybrid = HybridRetriever([bm25, dense], k=config.get("rrf_k", 60))

    return {
        "bm25": bm25,
        "dense": dense,
        "hybrid": hybrid,
    }
