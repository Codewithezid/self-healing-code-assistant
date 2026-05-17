from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class PDFIngestionConfig:
    """Runtime options for ingesting project PDFs into markdown/json."""

    input_paths: tuple[str, ...]
    output_dir: str
    format: str = "markdown,json"
    recursive: bool = True


@dataclass(frozen=True)
class PDFIngestionSummary:
    """Result metadata for a completed ingestion run."""

    inputs: list[str]
    output_dir: str
    format: str
    pdf_count: int


def ingest_pdfs(
    config: PDFIngestionConfig,
    *,
    convert_func: Callable[..., object] | None = None,
) -> PDFIngestionSummary:
    """Convert PDF files using OpenDataLoader and return run metadata."""
    input_roots = _resolve_inputs(config.input_paths)
    pdf_count = sum(_count_pdfs(path, recursive=config.recursive) for path in input_roots)
    if pdf_count == 0:
        raise ValueError(
            "No PDF files were found in the configured input path(s). "
            "Add PDF files first, then retry ingestion."
        )

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = convert_func or _load_converter()
    converter(
        [str(path) for path in input_roots],
        output_dir=str(output_dir),
        format=config.format,
    )
    return PDFIngestionSummary(
        inputs=[str(path) for path in input_roots],
        output_dir=str(output_dir),
        format=config.format,
        pdf_count=pdf_count,
    )


def _resolve_inputs(input_paths: tuple[str, ...]) -> list[Path]:
    resolved: list[Path] = []
    for raw in input_paths:
        stripped = raw.strip()
        if not stripped:
            continue
        path = Path(stripped).resolve()
        if not path.exists():
            continue
        resolved.append(path)
    return resolved


def _count_pdfs(path: Path, *, recursive: bool) -> int:
    if path.is_file():
        return 1 if path.suffix.lower() == ".pdf" else 0
    if not path.is_dir():
        return 0
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sum(1 for _ in path.glob(pattern))


def _load_converter() -> Callable[..., object]:
    try:
        import opendataloader_pdf
    except ImportError as exc:
        raise RuntimeError(
            "opendataloader-pdf is not installed. "
            "Install it with: pip install -U opendataloader-pdf"
        ) from exc
    return opendataloader_pdf.convert
