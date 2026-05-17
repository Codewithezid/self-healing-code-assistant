from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.ingest_project_pdfs import _build_parser, run


class IngestProjectPDFsScriptTests(unittest.TestCase):
    def test_run_executes_ingestion_and_optional_reindex(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--input",
                "data/pdfs",
                "--output-dir",
                "docs/pdf_ingested",
                "--format",
                "markdown,json",
                "--reindex-rag",
            ]
        )

        with patch("scripts.ingest_project_pdfs.ingest_pdfs") as ingest_mock, patch(
            "scripts.ingest_project_pdfs.get_settings"
        ) as settings_mock:
            ingest_mock.return_value = MagicMock(
                pdf_count=2,
                inputs=["data/pdfs"],
                output_dir="docs/pdf_ingested",
            )
            settings_mock.return_value = MagicMock(
                project_root=Path("."),
                rag_qdrant_path=Path("data/qdrant"),
                rag_collection_name="code-assistant-project",
                rag_embedding_model="mistral-embed",
                rag_retrieval_k=4,
                rag_chunk_size=1200,
                rag_chunk_overlap=200,
                rag_auto_index=False,
                corrective_rag_enabled=False,
                corrective_rag_model="mistral-small-latest",
                corrective_rag_mode="balanced",
                corrective_rag_min_score=3,
                corrective_rag_retry_k=6,
            )
            rag_cls = MagicMock()
            run(args, rag_cls=rag_cls)

            ingest_mock.assert_called_once()
            rag_cls.assert_called_once()
            rag_cls.return_value.index_project.assert_called_once_with(force=True)

    def test_run_without_reindex_does_not_touch_rag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        with patch("scripts.ingest_project_pdfs.ingest_pdfs") as ingest_mock:
            ingest_mock.return_value = MagicMock(
                pdf_count=1,
                inputs=["data/pdfs"],
                output_dir="docs/pdf_ingested",
            )
            run(args)

            ingest_mock.assert_called_once()
            config = ingest_mock.call_args.args[0]
            self.assertEqual(config.input_paths, ("data/pdfs",))


if __name__ == "__main__":
    unittest.main()
