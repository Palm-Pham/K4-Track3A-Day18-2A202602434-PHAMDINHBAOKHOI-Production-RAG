"""Comparable baseline: paragraph chunks and dense-only retrieval."""

from __future__ import annotations

import time

import config
from src.generation import generate_answer
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, load_test_set, save_report


def main():
    print("BASIC RAG BASELINE (paragraph + dense-only)", flush=True)
    started = time.perf_counter()
    documents = load_documents()
    chunks = [{"text": chunk.text, "metadata": chunk.metadata}
              for document in documents
              for chunk in chunk_basic(document["text"], metadata=document["metadata"])]
    search = DenseSearch()
    search.index(chunks, collection=config.NAIVE_COLLECTION)
    build_ms = (time.perf_counter() - started) * 1000
    print(f"{len(chunks)} paragraph chunks indexed", flush=True)
    questions, answers, contexts, truths = [], [], [], []
    timings = []
    test_set = load_test_set()
    for index, item in enumerate(test_set, 1):
        started = time.perf_counter()
        results = search.search(item["question"], top_k=3, collection=config.NAIVE_COLLECTION)
        search_ms = (time.perf_counter() - started) * 1000
        evidence = [f"[Nguồn: {result.metadata.get('source', 'unknown')}]\n{result.text}" for result in results]
        started = time.perf_counter()
        answer = generate_answer(item["question"], evidence)
        timings.append({"question": item["question"], "search_ms": search_ms,
                        "generation_ms": (time.perf_counter() - started) * 1000,
                        "generation_status": generate_answer.last_status})
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item["ground_truth"])
        print(f"[{index}/{len(test_set)}] {item['question']}", flush=True)
    started = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    results.setdefault("metadata", {}).update({
        "pipeline": "naive", "retrieval_embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
        "latency": {"build_ms": build_ms, "evaluation_ms": (time.perf_counter() - started) * 1000,
                    "queries": timings},
    })
    save_report(results, [], path=str(config.REPORTS_DIR / "naive_baseline_report.json"))
    print(f"Baseline evaluation status: {results.get('status', 'unknown')}", flush=True)
    return results


if __name__ == "__main__":
    main()
