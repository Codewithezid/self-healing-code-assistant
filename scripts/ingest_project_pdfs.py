from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.code_assistant.pdf_ingestion import PDFIngestionConfig, ingest_pdfs
from src.code_assistant.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert project PDF files into markdown/json with OpenDataLoader PDF "
            "and optionally refresh the project RAG index."
        )
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=None,
        help=(
            "PDF input path (file or directory). "
            "Repeat this flag to ingest multiple paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="docs/pdf_ingested",
        help="Directory for converted markdown/json output files.",
    )
    parser.add_argument(
        "--format",
        default="markdown,json",
        help="OpenDataLoader output format list.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Disable recursive PDF discovery inside input directories.",
    )
    parser.add_argument(
        "--reindex-rag",
        action="store_true",
        help="Rebuild the local project Qdrant index after ingestion completes.",
    )
    return parser


def run(args: argparse.Namespace, *, rag_cls=None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    config = PDFIngestionConfig(
        input_paths=tuple(args.inputs or ["data/pdfs"]),
        output_dir=args.output_dir,
        format=args.format,
        recursive=not args.non_recursive,
    )
    summary = ingest_pdfs(config)
    print(
        "PDF ingestion complete: "
        f"{summary.pdf_count} PDF(s) converted from {len(summary.inputs)} input path(s) "
        f"into {summary.output_dir}."
    )

    if args.reindex_rag:
        if rag_cls is None:
            from src.code_assistant.rag import ProjectRAG

            rag_cls = ProjectRAG

        settings = get_settings()
        rag = rag_cls(
            project_root=settings.project_root,
            qdrant_path=settings.rag_qdrant_path,
            collection_name=settings.rag_collection_name,
            embedding_model=settings.rag_embedding_model,
            retrieval_k=settings.rag_retrieval_k,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            auto_index=settings.rag_auto_index,
            corrective_enabled=settings.corrective_rag_enabled,
            corrective_model=settings.corrective_rag_model,
            corrective_mode=settings.corrective_rag_mode,  # type: ignore[arg-type]
            corrective_min_score=settings.corrective_rag_min_score,
            corrective_retry_k=settings.corrective_rag_retry_k,
        )
        stats = rag.index_project(force=True)
        print(
            "RAG index refreshed: "
            f"{stats['files']} file(s), {stats['chunks']} chunk(s) "
            f"at {settings.rag_qdrant_path.resolve()}."
        )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"PDF ingestion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
