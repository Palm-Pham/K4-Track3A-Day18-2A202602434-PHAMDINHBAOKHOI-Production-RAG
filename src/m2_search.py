from __future__ import annotations

"""Vietnamese lexical search, Qdrant dense search, and reciprocal rank fusion."""

import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import dataclass
from functools import lru_cache
from operator import index as integer_index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, HYBRID_TOP_K


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def _limit(top_k: int) -> int:
    try:
        return max(0, integer_index(top_k))
    except TypeError as exc:
        raise ValueError("top_k must be an integer") from exc


def segment_vietnamese(text: str) -> str:
    """Use the same word segmentation for documents and queries."""
    if not text.strip():
        return ""
    from underthesea import word_tokenize

    return word_tokenize(text, format="text").replace("_", " ")


def _tokens(text: str) -> list[str]:
    return segment_vietnamese(text).casefold().split()


def _identity(text: str, metadata: dict) -> str:
    """Identify chunks by source and location, rather than answer text alone."""
    source = metadata.get("source", "")
    if metadata.get("chunk_id") is not None:
        identity = [source, "chunk_id", str(metadata["chunk_id"])]
    else:
        identity = [source, metadata.get("section", ""),
                    metadata.get("chunk_index"), metadata.get("parent_id"), text]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                    sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _prepare_document(chunk: dict) -> dict:
    metadata = dict(chunk.get("metadata") or {})
    if chunk.get("parent_id") is not None:
        metadata.setdefault("parent_id", chunk["parent_id"])
    original_text = chunk.get("original_text", metadata.get("original_text", chunk["text"]))
    if not isinstance(chunk["text"], str) or not isinstance(original_text, str):
        raise TypeError("Chunk text and original_text must be strings")
    metadata.setdefault("chunk_id", _identity(original_text, metadata))
    return {"text": original_text, "retrieval_text": chunk["text"], "metadata": metadata}


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self.documents = [_prepare_document(chunk) for chunk in chunks]
        self.corpus_tokens = [_tokens(doc["retrieval_text"]) for doc in self.documents]
        self.bm25 = None
        if any(self.corpus_tokens):
            self.bm25 = BM25Okapi(self.corpus_tokens)
            # Okapi's negative average IDF on tiny corpora can suppress every
            # matching document. Keep matching common words above zero.
            for token, idf in self.bm25.idf.items():
                if idf <= 0:
                    self.bm25.idf[token] = 1e-6

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        limit = _limit(top_k)
        if self.bm25 is None or not query.strip() or not limit:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=dict(self.documents[i]["metadata"]), method="bm25")
                for i in indices if scores[i] > 0][:limit]


@lru_cache(maxsize=4)
def _load_encoder(model_name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    return SentenceTransformer(model_name, device=device)


class DenseSearch:
    def __init__(self, model_name: str | None = None, embedding_dim: int | None = None,
                 location: str | None = None, client=None):
        from qdrant_client import QdrantClient

        self.model_name = model_name or config.EMBEDDING_MODEL
        self.embedding_dim = embedding_dim if embedding_dim is not None else config.EMBEDDING_DIM
        if not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError("EMBEDDING_DIM must be a positive integer")
        configured_location = (getattr(config, "QDRANT_LOCATION", "")
                               if location is None else location)
        if client is not None:
            self.client = client
        elif configured_location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif configured_location:
            self.client = QdrantClient(path=configured_location)
        else:
            self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT,
                                       timeout=30)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                self._encoder = _load_encoder(self.model_name, getattr(config, "MODEL_DEVICE", "cpu"))
            except Exception as exc:
                raise RuntimeError(f"Cannot load embedding model {self.model_name!r}; "
                                   "download/cache this model before dense retrieval") from exc
        dimension = self._encoder.get_sentence_embedding_dimension()
        if dimension != self.embedding_dim:
            raise ValueError(f"Embedding model {self.model_name!r} produces {dimension} dimensions, "
                             f"but EMBEDDING_DIM={self.embedding_dim}")
        return self._encoder

    def _encode(self, texts):
        import numpy as np

        vectors = np.asarray(self._get_encoder().encode(
            texts, batch_size=getattr(config, "EMBEDDING_BATCH_SIZE", 16),
            show_progress_bar=False, normalize_embeddings=True), dtype=float)
        expected_shape = ((len(texts), self.embedding_dim) if isinstance(texts, list)
                          else (self.embedding_dim,))
        if vectors.shape != expected_shape or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid embeddings: expected shape {expected_shape} with finite "
                             f"values, received {vectors.shape}")
        return vectors

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Replace a lab collection only after validating all embeddings."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        documents = [_prepare_document(chunk) for chunk in chunks]
        vectors = self._encode([doc["retrieval_text"] for doc in documents]) if documents else []
        points = [PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, _identity(doc["text"], doc["metadata"]))),
            vector=vector.tolist(),
            payload={**doc["metadata"], "text": doc["text"],
                     "_retrieval_text": doc["retrieval_text"], "_embedding_model": self.model_name},
        ) for doc, vector in zip(documents, vectors)]
        try:
            if self.client.collection_exists(collection_name=collection):
                self.client.delete_collection(collection_name=collection)
            self.client.create_collection(collection_name=collection,
                                          vectors_config=VectorParams(
                                              size=self.embedding_dim, distance=Distance.COSINE))
            for offset in range(0, len(points), 128):
                self.client.upsert(collection_name=collection,
                                   points=points[offset:offset + 128], wait=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot index Qdrant collection {collection!r}; start Qdrant "
                               "or explicitly configure QDRANT_LOCATION=:memory:") from exc

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        limit = _limit(top_k)
        if not query.strip() or not limit:
            return []
        query_vector = self._encode(query).tolist()
        try:
            response = self.client.query_points(collection_name=collection, query=query_vector,
                                                limit=limit, with_payload=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot query Qdrant collection {collection!r}; "
                               "index it with the configured embedding model first") from exc
        results = []
        for point in response.points:
            payload = point.payload or {}
            stored_model = payload.get("_embedding_model")
            if stored_model and stored_model != self.model_name:
                raise ValueError(f"Collection {collection!r} contains embeddings from "
                                 f"{stored_model!r}, expected {self.model_name!r}; reindex it")
            if "text" not in payload:
                raise ValueError(f"Qdrant point {point.id!r} is missing source text")
            metadata = {key: value for key, value in payload.items()
                        if key not in {"text", "_retrieval_text", "_embedding_model"}}
            results.append(SearchResult(text=payload["text"], score=float(point.score),
                                        metadata=metadata, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Add each document's 1-based reciprocal rank once per retrieval method."""
    limit = _limit(top_k)
    if not math.isfinite(k) or k < 0:
        raise ValueError("RRF k must be finite and nonnegative")
    if not limit:
        return []
    fused = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results, start=1):
            identity = _identity(result.text, result.metadata)
            if identity in seen:
                continue
            seen.add(identity)
            if identity not in fused:
                fused[identity] = [0.0, result]
            fused[identity][0] += 1.0 / (k + rank)
    ordered = sorted(fused.values(), key=lambda item: item[0], reverse=True)
    return [SearchResult(text=result.text, score=score, metadata=dict(result.metadata),
                         method="hybrid") for score, result in ordered[:limit]]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        if not _limit(top_k) or not query.strip():
            return []
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    sample = "Nhân viên được nghỉ phép năm"
    print(f"Original:  {sample}")
    print(f"Segmented: {segment_vietnamese(sample)}")
