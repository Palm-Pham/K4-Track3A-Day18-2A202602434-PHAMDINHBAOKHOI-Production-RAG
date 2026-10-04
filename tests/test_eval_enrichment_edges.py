"""Evaluation/enrichment contracts verified without external API calls."""

import json
import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace

import pytest

from src import m4_eval as evaluation
from src import m5_enrichment as enrichment


def _fake_evaluator(monkeypatch, records=None, error=None):
    captured = {}

    class Dataset:
        @staticmethod
        def from_dict(data):
            captured["dataset"] = data
            return data

    class Client:
        def __init__(self, **kwargs):
            self.options = kwargs
            captured.setdefault("clients", []).append(kwargs)

    class Wrapper:
        def __init__(self, client, **kwargs):
            self.client = client

    def evaluate(dataset, **kwargs):
        captured.update(kwargs)
        if error:
            raise error
        return SimpleNamespace(to_pandas=lambda: SimpleNamespace(to_dict=lambda **kwargs: records))

    modules = {
        "datasets": {"Dataset": Dataset},
        "langchain_openai": {"ChatOpenAI": Client, "OpenAIEmbeddings": Client},
        "ragas": {"evaluate": evaluate},
        "ragas.embeddings": {"LangchainEmbeddingsWrapper": Wrapper},
        "ragas.llms": {"LangchainLLMWrapper": Wrapper},
        "ragas.metrics": {metric: metric for metric in evaluation.METRICS},
        "ragas.run_config": {"RunConfig": lambda **kwargs: SimpleNamespace(**kwargs)},
    }
    for name, contents in modules.items():
        module = ModuleType(name)
        module.__dict__.update(contents)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(evaluation, "EVAL_ENABLED", True)
    monkeypatch.setattr(evaluation, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(evaluation, "OPENAI_BASE_URL", "https://example.com/api/v1")
    return captured


def _fake_enrichment_client(monkeypatch, tmp_path, payload=None, error=None):
    captured = {"calls": []}

    def create(**kwargs):
        captured["calls"].append(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload))])

    class Client:
        def __init__(self, **kwargs):
            captured["connection"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            captured["closed"] = True

    module = ModuleType("openai")
    module.OpenAI = Client
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setattr(enrichment, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(enrichment, "OPENAI_BASE_URL", "https://example.com/api/v1")
    monkeypatch.setattr(enrichment, "ENRICHMENT_ENABLED", True)
    monkeypatch.setattr(enrichment, "ENRICHMENT_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(enrichment, "_API_DISABLED_ERROR", None)
    monkeypatch.setattr(enrichment, "_API_FAILURE_COUNT", 0)
    monkeypatch.setattr(enrichment, "_LOADED_CACHE_PATH", None)
    enrichment._CACHE.clear()
    return captured


def test_missing_eval_key_preserves_evidence_and_marks_unavailable(monkeypatch):
    monkeypatch.setattr(evaluation, "EVAL_ENABLED", True)
    monkeypatch.setattr(evaluation, "OPENAI_API_KEY", "")
    result = evaluation.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["status"] == "unavailable"
    assert all(result[metric] == 0.0 for metric in evaluation.METRICS)
    assert result["per_question"][0].contexts == ["C"]
    failure = evaluation.failure_analysis(result["per_question"], bottom_n=1)[0]
    assert failure["worst_metric"] == "evaluation"
    assert failure["error_tree"]["branch"] == "evaluation"


def test_eval_validates_input_lengths_and_context_shape():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluation.evaluate_ragas(["Q"], [], [["C"]], ["GT"])
    with pytest.raises(ValueError, match="list of text strings"):
        evaluation.evaluate_ragas(["Q"], ["A"], ["C"], ["GT"])


def test_real_ragas_contract_and_partial_scores(monkeypatch):
    captured = _fake_evaluator(monkeypatch, [
        {"faithfulness": 0.8, "answer_relevancy": float("nan"), "context_precision": 0.6, "context_recall": 0.7},
        {"faithfulness": 0.4, "answer_relevancy": 0.9, "context_precision": 0.8, "context_recall": 0.5},
    ])
    result = evaluation.evaluate_ragas(["Q1", "Q2"], ["A1", "A2"], [["C1"], ["C2"]], ["GT1", "GT2"])
    assert captured["dataset"]["ground_truth"] == ["GT1", "GT2"]
    assert len(captured["metrics"]) == 4
    assert captured["clients"][0]["base_url"] == "https://example.com/api/v1"
    assert captured["clients"][1]["check_embedding_ctx_length"] is False
    assert captured["run_config"].max_retries == 1
    assert result["status"] == "partial"
    assert result["faithfulness"] == pytest.approx(0.6)
    assert result["answer_relevancy"] == 0.9
    assert result["per_question"][0].missing_metrics == ["answer_relevancy"]


def test_eval_exception_reports_error_instead_of_fabricated_scores(monkeypatch):
    _fake_evaluator(monkeypatch, error=RuntimeError("test-key rejected"))
    result = evaluation.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    assert result["status"] == "unavailable"
    assert "test-key" not in result["error"]
    assert "[redacted]" in result["error"]


def test_failure_analysis_handles_dicts_and_preserves_error_tree():
    weak = evaluation.EvalResult("weak", "A", ["C"], "GT", 0.8, 0.7, 0.6, 0.1)
    strong = evaluation.EvalResult("strong", "A", ["C"], "GT", 0.9, 0.9, 0.9, 0.9)
    failures = evaluation.failure_analysis([asdict(strong), weak], bottom_n=1)
    assert failures[0]["question"] == "weak"
    assert failures[0]["worst_metric"] == "context_recall"
    assert failures[0]["contexts"] == ["C"]
    assert failures[0]["ground_truth"] == "GT"
    assert failures[0]["error_tree"]["branch"] == "retrieval"
    assert evaluation.failure_analysis([weak], bottom_n=0) == []


def test_report_serializes_dataclasses_and_unavailable_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluation, "EVAL_ENABLED", False)
    result = evaluation.evaluate_ragas(["Q"], ["A"], [["C"]], ["GT"])
    target = tmp_path / "nested" / "report.json"
    evaluation.save_report(result, evaluation.failure_analysis(result["per_question"]), str(target))
    report = json.loads(target.read_text())
    assert report["num_questions"] == 1
    assert report["per_question"][0]["answer"] == "A"
    assert report["status"] == "unavailable"
    assert report["metadata"]["scores_available"] is False


def test_combined_enrichment_one_call_cache_and_authoritative_metadata(monkeypatch, tmp_path):
    payload = json.dumps({"summary": "Nghỉ phép 15 ngày.", "questions": ["Được nghỉ phép bao nhiêu ngày?"],
                          "context": "Chính sách nghỉ phép hiện hành.",
                          "metadata": {"source": "invented.md", "parent_id": "wrong", "version": "1900", "topic": "nghỉ phép"}})
    captured = _fake_enrichment_client(monkeypatch, tmp_path, payload)
    chunk = {"text": "Nhân viên được nghỉ phép 15 ngày.",
             "metadata": {"source": "policy.md", "parent_id": "p1", "version": "2024"}}
    first = enrichment.enrich_chunks([chunk])[0]
    second = enrichment.enrich_chunks([chunk])[0]
    assert len(captured["calls"]) == 1
    assert captured["connection"]["base_url"] == "https://example.com/api/v1"
    assert captured["connection"]["max_retries"] == 0
    assert captured["calls"][0]["model"] == enrichment.LLM_MODEL
    assert captured["calls"][0]["response_format"] == {"type": "json_object"}
    assert first.original_text == chunk["text"]
    assert chunk["text"] in first.enriched_text
    assert "Được nghỉ phép bao nhiêu ngày?" in first.enriched_text
    assert first.auto_metadata["source"] == "policy.md"
    assert first.auto_metadata["version"] == "2024"
    assert first.auto_metadata["parent_id"] == "p1"
    assert first.auto_metadata["original_text"] == chunk["text"]
    assert second.auto_metadata["enrichment_status"] == "cached"


def test_bad_enrichment_response_falls_back_without_losing_text(monkeypatch, tmp_path):
    _fake_enrichment_client(monkeypatch, tmp_path, "this is not JSON")
    chunk = {"text": "Không được chia sẻ mật khẩu.", "parent_id": "p2", "metadata": {"source": "it.md"}}
    result = enrichment.enrich_chunks([chunk])[0]
    assert result.original_text == chunk["text"]
    assert chunk["text"] in result.enriched_text
    assert result.auto_metadata["parent_id"] == "p2"
    assert result.auto_metadata["enrichment_status"] == "fallback"
    assert "JSONDecodeError" in result.auto_metadata["enrichment_error"]


def test_enrichment_methods_are_honored(monkeypatch):
    monkeypatch.setattr(enrichment, "summarize_chunk", lambda text: "summary only")
    monkeypatch.setattr(enrichment, "generate_hypothesis_questions", lambda text: pytest.fail("HyQA should not run"))
    monkeypatch.setattr(enrichment, "contextual_prepend", lambda *args: pytest.fail("Contextual should not run"))
    monkeypatch.setattr(enrichment, "extract_metadata", lambda text: pytest.fail("Metadata should not run"))
    chunk = {"text": "original", "metadata": {"source": "policy.md"}}
    result = enrichment.enrich_chunks([chunk], methods=["summary"])[0]
    assert result.summary == "summary only"
    assert result.hypothesis_questions == []
    assert "summary only" in result.enriched_text
    assert enrichment.enrich_chunks([chunk], methods=[])[0].enriched_text == "original"
    with pytest.raises(ValueError, match="Unknown enrichment"):
        enrichment.enrich_chunks([chunk], methods=["unknown"])


def test_enrichment_auth_failure_stops_repeated_calls(monkeypatch, tmp_path):
    class AuthenticationError(Exception):
        status_code = 401

    captured = _fake_enrichment_client(monkeypatch, tmp_path, error=AuthenticationError("test-key rejected"))
    chunks = [{"text": "A", "metadata": {"source": "a.md"}}, {"text": "B", "metadata": {"source": "b.md"}}]
    result = enrichment.enrich_chunks(chunks)
    assert len(captured["calls"]) == 1
    assert all(item.auto_metadata["enrichment_status"] == "fallback" for item in result)
    assert all("test-key" not in item.auto_metadata.get("enrichment_error", "") for item in result)
