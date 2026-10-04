"""Run the baseline and production pipelines and write their comparison."""

import json
import time

import config

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main():
    from naive_baseline import main as run_baseline
    from src.pipeline import build_pipeline, evaluate_pipeline

    started = time.perf_counter()
    print("LAB 18: PRODUCTION RAG PIPELINE", flush=True)
    baseline = run_baseline()
    search, reranker = build_pipeline()
    production = evaluate_pipeline(search, reranker)
    measured = baseline.get("status") == "evaluated" and production.get("status") == "evaluated"
    print(f"\n{'Metric':<25} {'Naive':>9} {'Production':>12} {'Delta':>9}")
    comparison = {}
    for metric in METRICS:
        naive_score, prod_score = baseline.get(metric, 0), production.get(metric, 0)
        comparison[metric] = {"naive": naive_score, "production": prod_score,
                              "delta": prod_score - naive_score if measured else None}
        delta = f"{prod_score - naive_score:+.4f}" if measured else "N/A"
        print(f"{metric:<25} {naive_score:>9.4f} {prod_score:>12.4f} {delta:>9}")
    report = {"metrics": comparison, "measured": measured,
              "naive_status": baseline.get("status"), "production_status": production.get("status"),
              "total_seconds": time.perf_counter() - started}
    (config.REPORTS_DIR / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not measured:
        print("RAGAS unavailable or partial: numeric fallback values are not measured scores.")
    print(f"Reports saved in {config.REPORTS_DIR}; total {report['total_seconds']:.1f}s", flush=True)
    return report


if __name__ == "__main__":
    main()
