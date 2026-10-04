"""Boundary tests isolate model output while exercising real BM25 and Qdrant."""

import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from src import m2_search, m3_rerank
from src.m2_search import BM25Search, DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker, benchmark_reranker


def test_segment_uses_underthesea_and_removes_compound_underscores(monkeypatch):
    calls = []

    def word_tokenize(text, format):
        calls.append((text, format))
        return "Nhân_viên nghỉ_phép"

    monkeypatch.setitem(sys.modules, "underthesea", SimpleNamespace(word_tokenize=word_tokenize))
    assert m2_search.segment_vietnamese("Nhân viên nghỉ phép") == "Nhân viên nghỉ phép"
    assert calls == [("Nhân viên nghỉ phép", "text")]
    assert m2_search.segment_vietnamese("   ") == ""
    assert len(calls) == 1


def test_bm25_indexes_enrichment_but_returns_source_text(monkeypatch):
    monkeypatch.setattr(m2_search, "segment_vietnamese", lambda text: text)
    chunks = [{"text": "hypothesis leave days", "metadata": {
        "source": "handbook.md", "original_text": "12 days annually", "chunk_index": 0}}]
    search = BM25Search()
    search.index(chunks)
    results = search.search("HYPOTHESIS", top_k=10)
    assert len(results) == 1
    assert results[0].text == "12 days annually"
    assert results[0].metadata["source"] == "handbook.md"
    assert results[0].score > 0
    assert "chunk_id" not in chunks[0]["metadata"]
    assert search.search("missing") == []
    assert search.search("hypothesis", top_k=0) == []
    assert search.search("hypothesis", top_k=-2) == []
    search.index([])
    assert search.search("hypothesis") == []
    search.index([{"text": "  "}])
    assert search.search("hypothesis") == []


def test_rrf_exact_scores_and_source_identity():
    a = SearchResult("same answer", 99.0, {"source": "a.md", "chunk_id": "1"}, "bm25")
    b = SearchResult("same answer", 0.01, {"source": "b.md", "chunk_id": "1"}, "bm25")
    c = SearchResult("third", 5.0, {"source": "c.md", "chunk_id": "2"}, "dense")
    results = reciprocal_rank_fusion([[a, b], [b, c]], top_k=20)
    assert [item.metadata["source"] for item in results] == ["b.md", "a.md", "c.md"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)
    assert results[2].score == pytest.approx(1 / 62)
    assert all(item.method == "hybrid" for item in results)
    assert a.score == 99.0


def test_rrf_same_chunk_can_have_different_text_and_duplicate_list_entries():
    original = SearchResult("original", 1, {"source": "a", "chunk_id": "1"}, "bm25")
    enriched = SearchResult("enriched", 9, {"source": "a", "chunk_id": "1"}, "dense")
    results = reciprocal_rank_fusion([[original, original], [enriched]], top_k=1)
    assert results[0].text == "original"
    assert results[0].score == pytest.approx(2 / 61)
    assert reciprocal_rank_fusion([[original]], top_k=0) == []
    assert reciprocal_rank_fusion([[original]], top_k=-1) == []
    with pytest.raises(ValueError, match="nonnegative"):
        reciprocal_rank_fusion([[original]], k=-1)
    with pytest.raises(ValueError, match="integer"):
        reciprocal_rank_fusion([[original]], top_k=0.5)


class FakeEncoder:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if isinstance(texts, str):
            return np.array([1.0, 0.0, 0.0])
        return np.array([[1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.0]
                         for i in range(len(texts))])


@pytest.fixture
def dense():
    search = DenseSearch(model_name="test-encoder", embedding_dim=3, location=":memory:")
    search._encoder = FakeEncoder()
    yield search
    search.client.close()


def test_dense_real_qdrant_roundtrip_original_text_and_uuid(dense):
    chunks = [{"text": "enriched leave question", "metadata": {
        "source": "a.md", "chunk_id": "arbitrary-string", "original_text": "12 days"}},
        {"text": "password question", "metadata": {"source": "b.md", "original_text": "90 days"}}]
    dense.index(chunks, collection="edge-tests")
    results = dense.search("leave", top_k=1, collection="edge-tests")
    assert len(results) == 1
    assert results[0].text == "12 days"
    assert results[0].method == "dense"
    assert results[0].metadata["source"] == "a.md"
    assert results[0].metadata["chunk_id"] == "arbitrary-string"
    assert "_retrieval_text" not in results[0].metadata
    assert dense._encoder.calls[0][0] == ["enriched leave question", "password question"]
    assert dense._encoder.calls[0][1]["normalize_embeddings"] is True
    first_points, _ = dense.client.scroll(collection_name="edge-tests", limit=10)
    first_ids = {point.id for point in first_points}
    assert all(uuid.UUID(point_id) for point_id in first_ids)
    dense.index(chunks, collection="edge-tests")
    second_points, _ = dense.client.scroll(collection_name="edge-tests", limit=10)
    assert first_ids == {point.id for point in second_points}
    assert dense.search("leave", top_k=0, collection="missing") == []


def test_dense_dimension_validation_preserves_existing_index(dense):
    dense.index([{"text": "one"}], collection="edge-tests")
    dense._encoder = FakeEncoder(dimension=2)
    with pytest.raises(ValueError, match="produces 2 dimensions"):
        dense.index([{"text": "replacement"}], collection="edge-tests")
    assert dense.client.count(collection_name="edge-tests").count == 1


def test_dense_rejects_invalid_vector_shape_before_index_mutation(dense, monkeypatch):
    monkeypatch.setattr(dense._encoder, "encode", lambda *args, **kwargs: np.array([[float("nan")]]))
    with pytest.raises(ValueError, match="Invalid embeddings"):
        dense.index([{"text": "bad"}], collection="bad-vectors")
    assert not dense.client.collection_exists(collection_name="bad-vectors")


def test_dense_empty_index_has_no_model_calls(dense):
    dense.index([], collection="empty")
    assert dense.client.count(collection_name="empty").count == 0
    assert dense._encoder.calls == []


def test_dense_rejects_collection_embedded_with_different_model(dense):
    dense.index([{"text": "old"}], collection="edge-tests")
    dense.model_name = "other-encoder"
    with pytest.raises(ValueError, match="reindex"):
        dense.search("question", collection="edge-tests")


def test_dense_qdrant_failure_is_explicit(dense, monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(dense.client, "query_points", fail)
    with pytest.raises(RuntimeError, match="Cannot query Qdrant"):
        dense.search("question")


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_reranker_batch_score_and_metadata_mapping():
    model = FakeCrossEncoder(np.array([[-2.0], [4.0], [4.0]]))
    reranker = CrossEncoderReranker()
    reranker._model = model
    documents = [{"text": f"document {i}", "score": i / 10, "metadata": {"source": str(i)}}
                 for i in range(3)]
    results = reranker.rerank("question", documents, top_k=2)
    assert [result.text for result in results] == ["document 1", "document 2"]
    assert [result.original_score for result in results] == [0.1, 0.2]
    assert [result.rank for result in results] == [1, 2]
    assert [result.metadata["source"] for result in results] == ["1", "2"]
    assert model.calls[0][0] == [("question", doc["text"]) for doc in documents]
    assert len(model.calls) == 1
    results[0].metadata["source"] = "changed"
    assert documents[1]["metadata"]["source"] == "1"


def test_reranker_scalar_and_empty_documents():
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder(0.7)
    result = reranker.rerank("question", [{"text": "single"}])[0]
    assert result.rerank_score == pytest.approx(0.7)
    assert result.original_score == 0.0
    assert reranker.rerank("question", []) == []
    assert reranker.rerank("question", [{"text": "single"}], top_k=-1) == []
    assert len(reranker._model.calls) == 1


@pytest.mark.parametrize("scores", [[0.5], [0.5, float("nan")], [[1.0, 2.0], [3.0, 4.0]]])
def test_reranker_rejects_missing_nonfinite_or_multiclass_scores(scores):
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder(scores)
    with pytest.raises(ValueError, match="one finite relevance score"):
        reranker.rerank("question", [{"text": "first"}, {"text": "second"}])


def test_reranker_model_failure_is_explicit(monkeypatch):
    def fail(*args):
        raise OSError("missing model")

    monkeypatch.setattr(m3_rerank, "_load_cross_encoder", fail)
    with pytest.raises(RuntimeError, match="Cannot load cross-encoder"):
        CrossEncoderReranker().rerank("question", [{"text": "text"}])


def test_benchmark_rejects_zero_runs():
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_reranker(CrossEncoderReranker(), "question", [], n_runs=0)
