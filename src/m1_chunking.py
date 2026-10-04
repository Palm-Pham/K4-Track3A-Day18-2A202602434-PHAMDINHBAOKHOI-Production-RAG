"""Semantic, parent/child, and Markdown-aware chunking for the RAG pipeline.

Size arguments are character limits. Chunk boundaries retain source whitespace,
so joining a document's chunks reproduces its text without dropping content.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Read the text layer; an empty string explicitly identifies a scanned PDF."""
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()


def _ocr_pdf_text(path: str) -> str:
    """Opt-in OCR using locally installed Tesseract and PDFium, without downloads."""
    import pypdfium2
    import pytesseract

    pdf = pypdfium2.PdfDocument(path)
    try:
        languages = os.getenv("RAG_OCR_LANGUAGE", "vie+eng")
        pages = []
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = None
            try:
                bitmap = page.render(scale=2)
                pages.append(pytesseract.image_to_string(bitmap.to_pil(), lang=languages))
            finally:
                if bitmap is not None:
                    bitmap.close()
                page.close()
        return "\n\n".join(pages).strip()
    finally:
        pdf.close()


def _document_metadata(source: str, text: str) -> dict:
    """Extract declared policy attributes, retaining provenance for every chunk."""
    title = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    metadata = {"source": source, "title": title.group(1).strip() if title else Path(source).stem}
    fields = {
        "version": r"Phiên bản:\s*([^|\n]+)",
        "effective_date": r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})",
        "department": r"Phòng ban:\s*([^|\n]+)",
    }
    for key, pattern in fields.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if key == "effective_date":
                day, month, year = value.split("/")
                value = f"{year}-{month}-{day}"
            metadata[key] = value
    metadata["policy_id"] = re.sub(r"_v\d+(?:\.\d+)?$", "", Path(source).stem, flags=re.IGNORECASE)
    explicit_status = re.search(r"Trạng thái:\s*([^|\n]+)", text, flags=re.IGNORECASE)
    superseded = bool(explicit_status and re.search(r"ĐÃ THAY THẾ", explicit_status.group(1), re.I))
    superseded = superseded or bool(re.search(r"(?:này|1\.0) đã được thay thế", text, re.I))
    metadata["is_current"] = not superseded
    metadata["status"] = "superseded" if superseded else "current"
    return metadata


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load Markdown/text PDFs and explicitly report PDFs awaiting OCR.

    OCR runs only when RAG_OCR_PDFS=true and the optional local OCR dependencies
    are present. ``load_documents.last_report`` records skipped/failed PDFs for
    pipeline reports; a scanned document is never presented as successfully read.
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")
    report = {"loaded_documents": 0, "skipped_pdfs": [], "pdf_errors": [], "ocr_documents": []}
    docs = []
    ocr_enabled = os.getenv("RAG_OCR_PDFS", "false").strip().lower() in {"1", "true", "yes"}
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".md", ".pdf"})
    for path in paths:
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8-sig")
            extraction = "markdown"
        else:
            try:
                text = _extract_pdf_text(str(path))
                extraction = "pdf_text"
                if not text and ocr_enabled:
                    text = _ocr_pdf_text(str(path))
                    extraction = "pdf_ocr"
                    if text:
                        report["ocr_documents"].append(path.name)
            except Exception as exc:
                report["pdf_errors"].append({"source": path.name, "error": str(exc)})
                warnings.warn(f"Could not read PDF {path.name}: {exc}", RuntimeWarning, stacklevel=2)
                continue
            if not text:
                report["skipped_pdfs"].append({"source": path.name, "reason": "no text layer; OCR required"})
                warnings.warn(
                    f"Skipped scanned PDF {path.name}: no text layer; OCR required. "
                    "Set RAG_OCR_PDFS=true after installing local OCR dependencies.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
        if text.strip():
            metadata = _document_metadata(path.name, text)
            metadata["extraction_method"] = extraction
            docs.append({"text": text, "metadata": metadata})

    # Versions remain retrievable for historical queries; status also makes it
    # possible to select the current policy explicitly for present-tense queries.
    families: dict[str, list[dict]] = {}
    for doc in docs:
        families.setdefault(doc["metadata"]["policy_id"], []).append(doc)
    for family in families.values():
        if len(family) < 2:
            continue
        latest = max(family, key=lambda d: (
            d["metadata"].get("effective_date", ""),
            tuple(int(v) for v in re.findall(r"\d+", d["metadata"].get("version", "0"))),
        ))
        for doc in family:
            current = doc is latest and doc["metadata"]["is_current"]
            doc["metadata"].update(is_current=current, status="current" if current else "superseded")
    report["loaded_documents"] = len(docs)
    load_documents.last_report = report
    return docs


load_documents.last_report = {"loaded_documents": 0, "skipped_pdfs": [], "pdf_errors": [], "ocr_documents": []}


def _validate_size(size: int, name: str) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{name} must be a positive integer character limit")


def _bounded_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Prefer nearby paragraph/sentence/word boundaries, with a hard size cap."""
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\n", r"\s+"):
                boundaries = [m.end() for m in re.finditer(pattern, window) if m.end() >= size / 2]
                if boundaries:
                    end = start + boundaries[-1]
                    break
        spans.append((start, end))
        start = end
    return spans


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline chunking that prefers paragraph boundaries within a size limit."""
    _validate_size(chunk_size, "chunk_size")
    if not text.strip():
        return []
    return [Chunk(text[start:end], {
        **(metadata or {}), "chunk_index": index, "strategy": "basic",
        "start_char": start, "end_char": end,
    }) for index, (start, end) in enumerate(_bounded_spans(text, chunk_size))]


@lru_cache(maxsize=1)
def _get_semantic_model():
    """Load once, only when a document needs an embedding comparison."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2"), device=os.getenv("MODEL_DEVICE", "cpu"),
    )


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        if text[start:match.end()].strip():
            spans.append((start, match.end()))
            start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None, *,
                   max_chunk_size: int = HIERARCHICAL_PARENT_SIZE) -> list[Chunk]:
    """Group adjacent sentences by cosine similarity, preserving exact text.

    Related neighbors join when similarity is at least ``threshold``. Oversized
    groups/sentences use nearby boundaries to respect ``max_chunk_size``.
    """
    _validate_size(max_chunk_size, "max_chunk_size")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("threshold must be a finite cosine similarity in [-1, 1]")
    if not text.strip():
        return []
    spans = _sentence_spans(text)
    groups = []
    group_start = 0
    if len(spans) > 1:
        import numpy as np

        sentences = [text[start:end].strip() for start, end in spans]
        vectors = np.asarray(_get_semantic_model().encode(
            sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ))
        if vectors.ndim != 2 or len(vectors) != len(spans) or not np.isfinite(vectors).all():
            raise ValueError("Semantic encoder must return one embedding vector per sentence")
        for i in range(1, len(spans)):
            previous, current = vectors[i - 1], vectors[i]
            denominator = float(np.linalg.norm(previous) * np.linalg.norm(current))
            similarity = float(np.dot(previous, current) / denominator) if denominator else 0.0
            if similarity < threshold or spans[i][1] - group_start > max_chunk_size:
                groups.append((group_start, spans[i][0]))
                group_start = spans[i][0]
    groups.append((group_start, len(text)))
    chunks = []
    for group_start, group_end in groups:
        for start, end in _bounded_spans(text[group_start:group_end], max_chunk_size):
            start, end = start + group_start, end + group_start
            chunks.append(Chunk(text[start:end], {
                **(metadata or {}), "strategy": "semantic", "threshold": threshold,
                "chunk_index": len(chunks), "start_char": start, "end_char": end,
            }))
    return chunks


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """Index small children and use their stable parent IDs to expand context."""
    _validate_size(parent_size, "parent_size")
    _validate_size(child_size, "child_size")
    if child_size > parent_size:
        raise ValueError("child_size must not exceed parent_size")
    if not text.strip():
        return [], []
    metadata = metadata or {}
    identity = str(metadata.get("source", "document")) + "\0" + text
    document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    parents, children = [], []
    for parent_index, (start, end) in enumerate(_bounded_spans(text, parent_size)):
        parent_id = f"parent_{document_id}_{parent_index}"
        parent_text = text[start:end]
        parents.append(Chunk(parent_text, {
            **metadata, "strategy": "hierarchical", "chunk_type": "parent",
            "parent_id": parent_id, "chunk_id": parent_id, "chunk_index": parent_index,
            "start_char": start, "end_char": end,
        }))
        for child_index, (child_start, child_end) in enumerate(_bounded_spans(parent_text, child_size)):
            children.append(Chunk(parent_text[child_start:child_end], {
                **metadata, "strategy": "hierarchical", "chunk_type": "child",
                "parent_id": parent_id, "chunk_id": f"{parent_id}_child_{child_index}",
                "chunk_index": len(children), "child_index": child_index,
                "start_char": start + child_start, "end_char": start + child_end,
            }, parent_id=parent_id))
    return parents, children


def _markdown_headings(text: str) -> list[tuple[int, int, str, int]]:
    """Find ATX/Setext headings while ignoring header-like text in code fences."""
    lines = text.splitlines(keepends=True)
    headings = []
    offset = 0
    fence_char, fence_length = "", 0
    skip_underline = False
    for index, line in enumerate(lines):
        if skip_underline:
            offset += len(line)
            skip_underline = False
            continue
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            run, suffix = fence.groups()
            if not fence_char:
                fence_char, fence_length = run[0], len(run)
            elif run[0] == fence_char and len(run) >= fence_length and not suffix.strip():
                fence_char, fence_length = "", 0
            offset += len(line)
            continue
        if not fence_char:
            atx = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line.rstrip("\r\n"))
            if atx:
                marks, title = atx.groups()
                title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
                headings.append((offset, len(marks), title, offset + len(line)))
            elif line.strip() and index + 1 < len(lines):
                underline = re.match(r"^ {0,3}(=+|-+)[ \t]*$", lines[index + 1].rstrip("\r\n"))
                if underline and not re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s|>|\|)", line):
                    level = 1 if underline.group(1)[0] == "=" else 2
                    headings.append((offset, level, line.strip(), offset + len(line) + len(lines[index + 1])))
                    skip_underline = True
        offset += len(line)
    return headings


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk complete Markdown sections; preserve tables, lists, and code blocks."""
    if not text.strip():
        return []
    metadata = metadata or {}
    headings = _markdown_headings(text)
    boundaries = [(0, 0, "preamble", 0)] if not headings or headings[0][0] > 0 else []
    boundaries.extend(headings)
    chunks = []
    ancestors: list[tuple[int, str]] = []
    for index, (start, level, title, _) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(text)
        if level:
            while ancestors and ancestors[-1][0] >= level:
                ancestors.pop()
            ancestors.append((level, title))
        chunks.append(Chunk(text[start:end], {
            **metadata, "strategy": "structure", "section": title,
            "section_path": [ancestor[1] for ancestor in ancestors], "heading_level": level,
            "chunk_index": len(chunks), "start_char": start, "end_char": end,
        }))
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Compare strategy sizes across real document boundaries and metadata."""
    def stats(chunks):
        lengths = [len(chunk.text) for chunk in chunks]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {"count": len(lengths), "avg_len": round(sum(lengths) / len(lengths)),
                "min_len": min(lengths), "max_len": max(lengths)}

    basic, semantic, parents, children, structure = [], [], [], [], []
    for document in documents:
        text, metadata = document["text"], document.get("metadata", {})
        basic.extend(chunk_basic(text, metadata=metadata))
        semantic.extend(chunk_semantic(text, metadata=metadata))
        document_parents, document_children = chunk_hierarchical(text, metadata=metadata)
        parents.extend(document_parents)
        children.extend(document_children)
        structure.extend(chunk_structure_aware(text, metadata=metadata))
    results = {
        "basic": stats(basic), "semantic": stats(semantic),
        "hierarchical": {**stats(children), "parents": len(parents)}, "structure": stats(structure),
    }
    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, result in results.items():
        print(f"{name:<15} {result['count']:>7} {result['avg_len']:>5} {result['min_len']:>5} {result['max_len']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    compare_strategies(docs)
