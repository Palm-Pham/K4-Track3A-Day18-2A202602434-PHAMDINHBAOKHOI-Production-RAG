from __future__ import annotations

"""Batch cross-encoder reranking with reusable model instances."""

import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import RERANK_TOP_K
from src.m2_search import _limit


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str, max_length: int, device: str):
    import torch
    from sentence_transformers import CrossEncoder

    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    return CrossEncoder(model_name, max_length=max_length, device=device)


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or getattr(config, "RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                self._model = _load_cross_encoder(
                    self.model_name, getattr(config, "RERANK_MAX_LENGTH", 512),
                    getattr(config, "MODEL_DEVICE", "cpu"))
            except Exception as exc:
                raise RuntimeError(f"Cannot load cross-encoder {self.model_name!r}; "
                                   "download/cache this model before reranking") from exc
        return self._model

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        import numpy as np

        limit = _limit(top_k)
        if not documents or not limit:
            return []
        pairs = [(query, doc["text"]) for doc in documents]
        scores = np.asarray(self._load_model().predict(
            pairs, batch_size=getattr(config, "RERANK_BATCH_SIZE", 16),
            show_progress_bar=False, convert_to_numpy=True), dtype=float)
        if scores.ndim == 0 and len(documents) == 1:
            scores = scores.reshape(1)
        elif scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder must return one finite relevance score per document; "
                             f"received shape {scores.shape} for {len(documents)} documents")
        indices = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), i))[:limit]
        return [RerankResult(text=documents[i]["text"],
                             original_score=float(documents[i].get("score", 0.0)),
                             rerank_score=float(scores[i]),
                             metadata=dict(documents[i].get("metadata") or {}), rank=rank)
                for rank, i in enumerate(indices, start=1)]


class FlashrankReranker:
    """Optional lightweight reranker, explicitly selected by the caller."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict],
               top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        limit = _limit(top_k)
        if not documents or not limit:
            return []
        from flashrank import Ranker, RerankRequest

        if self._model is None:
            self._model = Ranker()
        passages = [{"id": i, "text": doc["text"]} for i, doc in enumerate(documents)]
        scored = self._model.rerank(RerankRequest(query=query, passages=passages))
        return [RerankResult(text=documents[int(result["id"])]["text"],
                             original_score=float(documents[int(result["id"])].get("score", 0.0)),
                             rerank_score=float(result["score"]),
                             metadata=dict(documents[int(result["id"])].get("metadata") or {}),
                             rank=rank)
                for rank, result in enumerate(scored[:limit], start=1)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Measure actual inference calls; the first call may include model loading."""
    if not isinstance(n_runs, int) or n_runs <= 0:
        raise ValueError("n_runs must be a positive integer")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    for result in CrossEncoderReranker().rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
