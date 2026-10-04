"""Four RAGAS metrics, honest unavailable results, and diagnostic reports."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TEST_SET_PATH = config.TEST_SET_PATH
OPENAI_API_KEY = getattr(config, "OPENAI_API_KEY", "")
OPENAI_BASE_URL = getattr(config, "OPENAI_BASE_URL", "")
RAGAS_MODEL = getattr(config, "RAGAS_MODEL", "gpt-4o-mini")
RAGAS_EMBEDDING_MODEL = getattr(config, "RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
API_TIMEOUT = getattr(config, "API_TIMEOUT", 60)
EVAL_ENABLED = getattr(config, "EVAL_ENABLED", True)
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    status: str = "evaluated"
    error: str | None = None
    missing_metrics: list[str] = field(default_factory=list)


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _has_key() -> bool:
    key = str(OPENAI_API_KEY or "").strip()
    return bool(key) and key.lower() not in {"sk-...", "your-api-key", "your_api_key", "changeme"}


def _safe_error(error: Exception | str) -> str:
    message = f"{type(error).__name__}: {error}" if isinstance(error, Exception) else error
    return str(message).replace(OPENAI_API_KEY, "[redacted]") if OPENAI_API_KEY else str(message)


def _metric_value(value) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) and 0 <= numeric <= 1 else None


def _unavailable(questions, answers, contexts, ground_truths, error: str, metadata: dict) -> dict:
    rows = [
        EvalResult(question, answer, list(context), ground_truth, 0.0, 0.0, 0.0, 0.0,
                   status="unavailable", error=error, missing_metrics=list(METRICS))
        for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths)
    ]
    return {**dict.fromkeys(METRICS, 0.0), "per_question": rows,
            "status": "unavailable", "error": error,
            "metadata": {**metadata, "scores_available": False,
                         "score_note": "Zero placeholders; no RAGAS scores were measured."}}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Evaluate using RAGAS 0.1.x; missing scores are explicitly marked unavailable.

    Aggregate means exclude failed metric samples. Numeric zero placeholders satisfy
    the report interface, while status/missing_metrics prevent treating them as scores.
    """
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if any(not isinstance(context, (list, tuple)) or
           any(not isinstance(text, str) for text in context) for context in contexts):
        raise ValueError("contexts must contain one list of text strings per question")

    started = time.perf_counter()
    metadata = {"evaluator": "ragas", "judge_model": RAGAS_MODEL,
                "embedding_model": RAGAS_EMBEDDING_MODEL,
                "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
                "num_questions": len(questions)}
    if not questions:
        return _unavailable(questions, answers, contexts, ground_truths, "No evaluation questions supplied.", metadata)
    if not EVAL_ENABLED:
        return _unavailable(questions, answers, contexts, ground_truths, "RAGAS evaluation disabled by configuration.", metadata)
    if not _has_key():
        return _unavailable(questions, answers, contexts, ground_truths, "Missing valid API key for RAGAS evaluation.", metadata)

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        run_config = RunConfig(timeout=int(API_TIMEOUT), max_retries=1, max_wait=1, max_workers=4)
        connection = {"api_key": OPENAI_API_KEY, "timeout": API_TIMEOUT, "max_retries": 0}
        if OPENAI_BASE_URL:
            connection["base_url"] = OPENAI_BASE_URL
        judge = ChatOpenAI(model=RAGAS_MODEL, temperature=0, **connection)
        # Send raw text so older LangChain does not tokenize an OpenRouter-prefixed name.
        embeddings = OpenAIEmbeddings(model=RAGAS_EMBEDDING_MODEL,
                                      check_embedding_ctx_length=False, **connection)
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": [list(context) for context in contexts],
                                     "ground_truth": ground_truths})
        result = evaluate(dataset,
                          metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=LangchainLLMWrapper(judge, run_config=run_config),
                          embeddings=LangchainEmbeddingsWrapper(embeddings),
                          run_config=run_config, raise_exceptions=False)
        records = result.to_pandas().to_dict(orient="records")
        if len(records) != len(questions):
            raise ValueError("RAGAS returned a different number of evaluation rows")
        per_question, measured = [], {metric: [] for metric in METRICS}
        for index, record in enumerate(records):
            scores, missing = {}, []
            for metric in METRICS:
                value = _metric_value(record.get(metric))
                if value is None:
                    scores[metric] = 0.0
                    missing.append(metric)
                else:
                    scores[metric] = value
                    measured[metric].append(value)
            status = "unavailable" if len(missing) == len(METRICS) else "partial" if missing else "evaluated"
            per_question.append(EvalResult(questions[index], answers[index], list(contexts[index]),
                                           ground_truths[index], **scores, status=status,
                                           error="RAGAS did not produce all finite metric scores." if missing else None,
                                           missing_metrics=missing))
        count = sum(len(values) for values in measured.values())
        status = "unavailable" if count == 0 else "partial" if count < len(questions) * len(METRICS) else "evaluated"
        metadata.update({"duration_seconds": time.perf_counter() - started,
                         "scores_available": count > 0,
                         "valid_samples_per_metric": {metric: len(values) for metric, values in measured.items()},
                         "score_note": "Means exclude unavailable samples; missing row metrics use zero placeholders."})
        return {**{metric: sum(values) / len(values) if values else 0.0 for metric, values in measured.items()},
                "per_question": per_question, "status": status,
                "error": "Some RAGAS metric scores are unavailable." if status != "evaluated" else None,
                "metadata": metadata}
    except Exception as error:  # noqa: BLE001 -- third-party evaluator failures must preserve the report
        metadata["duration_seconds"] = time.perf_counter() - started
        message = _safe_error(error)
        print(f"  RAGAS evaluation unavailable: {message}", flush=True)
        return _unavailable(questions, answers, contexts, ground_truths, message, metadata)


def failure_analysis(eval_results: list[EvalResult | dict], bottom_n: int = 10) -> list[dict]:
    """Rank weakest questions and map observed failures to the Diagnostic Tree."""
    diagnostics = {
        "faithfulness": ("Answer includes claims unsupported by the retrieved context.",
                         "Require citations for every claim, lower temperature, and abstain when evidence is absent.", "generation"),
        "answer_relevancy": ("Answer does not directly address the question.",
                             "Improve the answer prompt and preserve the question's constraints and requested scope.", "generation"),
        "context_precision": ("Retrieved context contains irrelevant or obsolete chunks.",
                              "Rerank candidates, filter superseded versions, and reduce the final context count.", "retrieval"),
        "context_recall": ("Retrieved context misses evidence needed for the reference answer.",
                           "Use hybrid BM25+dense retrieval, improve chunk boundaries, and expand matching child chunks to parents.", "retrieval"),
    }
    failures = []
    for result in eval_results:
        row = asdict(result) if is_dataclass(result) else dict(result)
        scores = {metric: _metric_value(row.get(metric)) for metric in METRICS}
        unavailable = row.get("status") == "unavailable" or all(value is None for value in scores.values())
        available_scores = {metric: value for metric, value in scores.items()
                            if value is not None and metric not in row.get("missing_metrics", [])}
        average = sum(available_scores.values()) / len(available_scores) if available_scores else 0.0
        worst = min(available_scores, key=available_scores.get) if available_scores and not unavailable else "evaluation"
        if worst == "evaluation":
            diagnosis = "Evaluation unavailable; zero placeholders cannot establish a retrieval or generation failure."
            fix = "Resolve the evaluator API/configuration error and rerun all four RAGAS metrics."
            branch = "evaluation"
        else:
            diagnosis, fix, branch = diagnostics[worst]
        failures.append({
            "question": row.get("question", ""), "answer": row.get("answer", ""),
            "contexts": row.get("contexts", []), "ground_truth": row.get("ground_truth", ""),
            "scores": {metric: value if value is not None else 0.0 for metric, value in scores.items()},
            "average_score": average, "worst_metric": worst,
            "score": available_scores.get(worst, 0.0), "diagnosis": diagnosis, "suggested_fix": fix,
            "status": "unavailable" if unavailable else row.get("status", "evaluated"), "error": row.get("error"),
            "missing_metrics": row.get("missing_metrics", []),
            "error_tree": {"root": "RAG quality", "branch": branch, "metric": worst,
                           "observation": diagnosis, "action": fix,
                           "path": ["RAG quality", branch, worst]},
        })
    return sorted(failures, key=lambda row: row["average_score"])[:max(0, bottom_n)]


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Persist aggregate and per-question evidence as strict, portable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = results.get("per_question", [])
    report = {"aggregate": {metric: results.get(metric, 0.0) for metric in METRICS},
              "num_questions": len(rows), "status": results.get("status", "evaluated"),
              "error": results.get("error"), "metadata": results.get("metadata", {}),
              "per_question": rows, "failures": failures}
    destination.write_text(json.dumps(_json_value(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report saved to {destination}", flush=True)


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
    print("Run pipeline.py to generate answers and evaluate them.")
