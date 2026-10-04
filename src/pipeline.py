"""Production pipeline: child retrieval, reranking, parent context, grounded answers."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.generation import generate_answer
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build the real hybrid index and keep the parent store for context expansion."""
    timings = {}
    started = time.perf_counter()
    print("PRODUCTION RAG PIPELINE", flush=True)
    documents = load_documents()
    if not documents:
        raise ValueError("No usable documents found in data/")
    parents_by_id = {}
    chunks = []
    for document in documents:
        parents, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        parents_by_id.update({parent.metadata["parent_id"]: parent for parent in parents})
        for child in children:
            metadata = {**child.metadata, "parent_id": child.parent_id,
                        "original_text": child.text}
            chunks.append({"text": child.text, "metadata": metadata})
    timings["load_and_chunk_ms"] = (time.perf_counter() - started) * 1000
    print(f"[M1] {len(documents)} documents, {len(parents_by_id)} parents, {len(chunks)} children", flush=True)

    started = time.perf_counter()
    enriched = enrich_chunks(chunks)
    indexed_chunks = [{"text": item.enriched_text,
                       "metadata": {**item.auto_metadata, "original_text": item.original_text}}
                      for item in enriched]
    timings["enrichment_ms"] = (time.perf_counter() - started) * 1000
    print(f"[M5] {len(indexed_chunks)} enriched chunks", flush=True)

    started = time.perf_counter()
    search = HybridSearch()
    search.index(indexed_chunks)
    search.parents = parents_by_id
    search.documents = documents
    search.ingestion_report = getattr(load_documents, "last_report", {})
    timings["index_ms"] = (time.perf_counter() - started) * 1000
    print("[M2] BM25 + Qdrant indexed", flush=True)

    started = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker._load_model()
    timings["reranker_load_ms"] = (time.perf_counter() - started) * 1000
    search.build_timings = timings
    search.query_timings = []
    print("[M3] CrossEncoder ready", flush=True)
    return search, reranker


def _asks_for_history(query: str) -> bool:
    return bool(re.search(
        r"phiên bản cũ|chính sách cũ|lịch sử|năm 202[23]|v2023|v1(?:\.0)?\b|old policy|historical",
        query.lower(),
    ))


def _context(text: str, metadata: dict) -> str:
    source = metadata.get("source", "unknown")
    status = metadata.get("status", "")
    version = metadata.get("version", "")
    label = f"Nguồn: {source}"
    if version:
        label += f" | Phiên bản: {version}"
    if status:
        label += f" | Trạng thái: {status}"
    return f"[{label}]\n{text}"


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Retrieve small children and send distinct, authoritative parents to the LLM."""
    times = {"question": query}
    started = time.perf_counter()
    results = search.search(query)
    if not _asks_for_history(query):
        results = [result for result in results
                   if result.metadata.get("is_current", True)
                   and result.metadata.get("status") != "superseded"]
    times["search_ms"] = (time.perf_counter() - started) * 1000
    documents = [{"text": result.text, "score": result.score, "metadata": result.metadata}
                 for result in results]
    started = time.perf_counter()
    # Rerank every candidate before deduplicating parents, so matching children
    # from the same document cannot consume all three context slots.
    reranked = reranker.rerank(query, documents, top_k=len(documents))
    times["rerank_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    contexts = []
    seen = set()
    for result in reranked:
        if config.RERANK_TOP_K <= 0:
            break
        parent_id = result.metadata.get("parent_id")
        parent = getattr(search, "parents", {}).get(parent_id)
        identity = parent_id or (result.metadata.get("source"), result.text)
        if identity in seen:
            continue
        seen.add(identity)
        contexts.append(_context(parent.text, parent.metadata) if parent
                        else _context(result.text, result.metadata))
        if len(contexts) >= config.RERANK_TOP_K:
            break
    times["parent_expansion_ms"] = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    answer = generate_answer(query, contexts)
    times["generation_ms"] = (time.perf_counter() - started) * 1000
    times["generation_status"] = generate_answer.last_status
    if hasattr(search, "query_timings"):
        search.query_timings.append(times)
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Answer benchmark questions without access to ground truth, then evaluate."""
    test_set = load_test_set()
    questions, answers, contexts, truths = [], [], [], []
    search.query_timings = []
    for index, item in enumerate(test_set, 1):
        answer, evidence = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item["ground_truth"])
        print(f"[{index}/{len(test_set)}] {item['question']}", flush=True)
    started = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    evaluation_ms = (time.perf_counter() - started) * 1000
    results.setdefault("metadata", {}).update({
        "pipeline": "production", "retrieval_embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANK_MODEL, "llm_model": config.LLM_MODEL,
        "parent_expansion": True, "current_policy_filter": True,
        "ingestion": getattr(search, "ingestion_report", {}),
        "latency": {"build_ms": search.build_timings, "evaluation_ms": evaluation_ms,
                    "queries": search.query_timings},
    })
    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures, path=str(config.REPORTS_DIR / "ragas_report.json"))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "latency_report.json").write_text(
        json.dumps(results["metadata"]["latency"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluation status: {results.get('status', 'unknown')}", flush=True)
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        print(f"  {metric}: {results.get(metric, 0):.4f}")
    return results


if __name__ == "__main__":
    started = time.perf_counter()
    retrieval, ranking = build_pipeline()
    evaluate_pipeline(retrieval, ranking)
    print(f"Total: {time.perf_counter() - started:.1f}s")
