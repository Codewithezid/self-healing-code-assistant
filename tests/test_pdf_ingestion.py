from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.code_assistant.pdf_ingestion import PDFIngestionConfig, ingest_pdfs


class _ConvertSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, input_path: list[str], **kwargs: str) -> None:
        self.calls.append((input_path, kwargs))


class PDFIngestionTests(unittest.TestCase):
    def test_ingest_calls_converter_with_expected_arguments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "data" / "pdfs"
            input_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "sample.pdf").write_bytes(b"%PDF-1.4")

            output_dir = root / "docs" / "pdf_ingested"
            spy = _ConvertSpy()
            config = PDFIngestionConfig(
                input_paths=(str(input_dir),),
                output_dir=str(output_dir),
                format="markdown,json",
                recursive=True,
            )

            summary = ingest_pdfs(config, convert_func=spy)

            self.assertEqual(summary.pdf_count, 1)
            self.assertEqual(summary.inputs, [str(input_dir.resolve())])
            self.assertEqual(summary.output_dir, str(output_dir.resolve()))
            self.assertEqual(len(spy.calls), 1)
            called_inputs, called_kwargs = spy.calls[0]
            self.assertEqual(called_inputs, [str(input_dir.resolve())])
            self.assertEqual(called_kwargs["output_dir"], str(output_dir.resolve()))
            self.assertEqual(called_kwargs["format"], "markdown,json")

    def test_ingest_fails_when_no_pdf_inputs_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "does-not-exist"
            config = PDFIngestionConfig(
                input_paths=(str(missing),),
                output_dir=str(Path(temp_dir) / "output"),
            )
            with self.assertRaisesRegex(ValueError, "No PDF files were found"):
                ingest_pdfs(config, convert_func=_ConvertSpy())


if __name__ == "__main__":
    unittest.main()
