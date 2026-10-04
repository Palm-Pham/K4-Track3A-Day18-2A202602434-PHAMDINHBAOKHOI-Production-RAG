"""Content preservation and document identity checks beyond the lab smoke tests."""

import numpy as np
import pytest

from src import m1_chunking as chunking


class FixedSentenceModel:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors)
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append(sentences)
        return self.vectors


def test_semantic_splits_at_actual_topic_change_without_losing_whitespace(monkeypatch):
    model = FixedSentenceModel([[4, 0], [2, 0], [0, 3]])
    monkeypatch.setattr(chunking, "_get_semantic_model", lambda: model)
    text = "  Cats rest.\nCats sleep.  Networks fail.\n"
    result = chunking.chunk_semantic(text, threshold=0.8, metadata={"source": "guide.md"})
    assert len(result) == 2
    assert result[0].text == "  Cats rest.\nCats sleep.  "
    assert result[1].text == "Networks fail.\n"
    assert "".join(chunk.text for chunk in result) == text
    assert model.calls == [["Cats rest.", "Cats sleep.", "Networks fail."]]
    assert all(chunk.metadata["source"] == "guide.md" for chunk in result)


def test_single_sentence_and_empty_input_do_not_load_model(monkeypatch):
    def fail_model_load():
        raise AssertionError("No embeddings needed")

    monkeypatch.setattr(chunking, "_get_semantic_model", fail_model_load)
    assert chunking.chunk_semantic(" \n ") == []
    text = "Nghỉ phép rất dài " * 30
    chunks = chunking.chunk_semantic(text, max_chunk_size=63)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(len(chunk.text) <= 63 for chunk in chunks)


def test_hierarchy_enforces_limits_and_reconstructs_each_parent():
    text = "  # Nội dung\n\n" + "đ" * 211 + "\n\n" + "Tài liệu tiếng Việt. " * 20
    parents, children = chunking.chunk_hierarchical(text, parent_size=120, child_size=37,
                                                  metadata={"source": "first.md", "version": "2.0"})
    assert "".join(parent.text for parent in parents) == text
    assert all(0 < len(parent.text) <= 120 for parent in parents)
    assert all(0 < len(child.text) <= 37 for child in children)
    assert len({child.metadata["chunk_id"] for child in children}) == len(children)
    for parent in parents:
        matched = [child for child in children if child.parent_id == parent.metadata["parent_id"]]
        assert "".join(child.text for child in matched) == parent.text
        for child in matched:
            assert child.metadata["parent_id"] == child.parent_id
            assert child.text == text[child.metadata["start_char"]:child.metadata["end_char"]]
            assert child.metadata["version"] == "2.0"


def test_parent_ids_stable_and_distinct_across_documents():
    first, _ = chunking.chunk_hierarchical("Shared content.", metadata={"source": "first.md"})
    again, _ = chunking.chunk_hierarchical("Shared content.", metadata={"source": "first.md"})
    second, _ = chunking.chunk_hierarchical("Shared content.", metadata={"source": "second.md"})
    changed, _ = chunking.chunk_hierarchical("Changed content.", metadata={"source": "first.md"})
    assert first[0].metadata["parent_id"] == again[0].metadata["parent_id"]
    assert len({chunks[0].metadata["parent_id"] for chunks in (first, second, changed)}) == 3


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_chunk_sizes_raise(size):
    with pytest.raises(ValueError):
        chunking.chunk_hierarchical("A", parent_size=size)
    with pytest.raises(ValueError):
        chunking.chunk_basic("A", chunk_size=size)


def test_structure_does_not_treat_code_as_headers_or_split_table():
    text = ("Preamble\n\n# Guide\n\n## API\n```python\n# code comment\n## another comment\n```\n"
            "\n| Name | Value |\n| --- | --- |\n| a | 3 |\n\n### Details\n- One\n- Two\n")
    result = chunking.chunk_structure_aware(text, metadata={"source": "api.md"})
    assert "".join(chunk.text for chunk in result) == text
    assert [chunk.metadata["section"] for chunk in result] == ["preamble", "Guide", "API", "Details"]
    api = result[2]
    assert "# code comment\n## another comment" in api.text
    assert "| Name | Value |\n| --- | --- |\n| a | 3 |" in api.text
    assert result[3].metadata["section_path"] == ["Guide", "API", "Details"]


def test_structure_supports_setext_headers_and_six_levels():
    text = "Title\n=====\nIntro.\n\nSubtitle\n--------\nBody.\n\n###### Deep ###\nMore.\n"
    chunks = chunking.chunk_structure_aware(text)
    assert "".join(chunk.text for chunk in chunks) == text
    assert [chunk.metadata["section"] for chunk in chunks] == ["Title", "Subtitle", "Deep"]
    assert [chunk.metadata["heading_level"] for chunk in chunks] == [1, 2, 6]


def test_load_documents_marks_versions_without_removing_history(tmp_path):
    for year, version in [(2023, "1.0"), (2024, "2.0")]:
        (tmp_path / f"nghi_phep_v{year}.md").write_text(
            f"# Nghỉ phép\n> Phiên bản: {version} | Ngày hiệu lực: 01/01/{year} | Phòng ban: Nhân sự\n",
            encoding="utf-8",
        )
    docs = chunking.load_documents(str(tmp_path))
    assert len(docs) == 2
    old, new = [doc["metadata"] for doc in docs]
    assert old["policy_id"] == new["policy_id"] == "nghi_phep"
    assert old["status"] == "superseded" and old["is_current"] is False
    assert new["status"] == "current" and new["effective_date"] == "2024-01-01"


def test_scanned_pdf_is_explicitly_reported_and_ocr_is_opt_in(tmp_path, monkeypatch):
    (tmp_path / "scan.pdf").write_bytes(b"fake scanned PDF")
    monkeypatch.setattr(chunking, "_extract_pdf_text", lambda path: "")
    monkeypatch.delenv("RAG_OCR_PDFS", raising=False)
    monkeypatch.setattr(chunking, "_ocr_pdf_text", lambda path: pytest.fail("OCR should be opt-in"))
    with pytest.warns(RuntimeWarning, match="Skipped scanned PDF scan.pdf"):
        assert chunking.load_documents(str(tmp_path)) == []
    assert chunking.load_documents.last_report["skipped_pdfs"] == [
        {"source": "scan.pdf", "reason": "no text layer; OCR required"},
    ]


def test_opt_in_ocr_records_extraction_method(tmp_path, monkeypatch):
    (tmp_path / "scan.pdf").write_bytes(b"fake scanned PDF")
    monkeypatch.setattr(chunking, "_extract_pdf_text", lambda path: "")
    monkeypatch.setattr(chunking, "_ocr_pdf_text", lambda path: "Recovered PDF text")
    monkeypatch.setenv("RAG_OCR_PDFS", "true")
    docs = chunking.load_documents(str(tmp_path))
    assert docs[0]["metadata"]["extraction_method"] == "pdf_ocr"
    assert chunking.load_documents.last_report["ocr_documents"] == ["scan.pdf"]
