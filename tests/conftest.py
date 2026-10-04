"""Unit tests never spend API credits; integration runs use the real provider."""

import pytest


@pytest.fixture(autouse=True)
def disable_external_llm_calls(monkeypatch):
    import config
    import src.m4_eval as evaluation
    import src.m5_enrichment as enrichment

    for module in (config, evaluation):
        monkeypatch.setattr(module, "EVAL_ENABLED", False, raising=False)
    for module in (config, enrichment):
        monkeypatch.setattr(module, "ENRICHMENT_ENABLED", False, raising=False)
