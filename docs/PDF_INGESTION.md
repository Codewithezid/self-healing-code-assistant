# PDF Ingestion Guide

This project integrates [OpenDataLoader PDF](https://github.com/opendataloader-project/opendataloader-pdf) to ingest external PDFs into the existing project RAG pipeline.

## What it does

1. Finds PDF files from one or more input paths.
2. Converts them to Markdown and JSON.
3. Writes outputs to a docs directory that is already indexed by project RAG.
4. Optionally rebuilds the local Qdrant index immediately.

## Command

Basic:

```powershell
python scripts/ingest_project_pdfs.py --reindex-rag
```

Custom paths:

```powershell
python scripts/ingest_project_pdfs.py `
  --input data/pdfs `
  --input archive/customer_docs `
  --output-dir docs/pdf_ingested `
  --format markdown,json `
  --reindex-rag
```

Non-recursive directory scan:

```powershell
python scripts/ingest_project_pdfs.py --input data/pdfs --non-recursive
```

## Output locations

- Input default: `data/pdfs/`
- Converted files default: `docs/pdf_ingested/`
- RAG index location: `data/qdrant/` (controlled by existing RAG settings)

## End-to-end example

Input:

- `data/pdfs/policy.pdf`
- `data/pdfs/reports/q1.pdf`

Command:

```powershell
python scripts/ingest_project_pdfs.py --reindex-rag
```

Output:

```text
PDF ingestion complete: 2 PDF(s) converted from 1 input path(s) into C:\...\docs\pdf_ingested.
RAG index refreshed: 42 file(s), 588 chunk(s) at C:\...\data\qdrant.
```

## Error handling

The command exits with status code `1` on failure and prints a root-cause message, for example:

```text
PDF ingestion failed: No PDF files were found in the configured input path(s). Add PDF files first, then retry ingestion.
```

If OpenDataLoader is missing:

```text
PDF ingestion failed: opendataloader-pdf is not installed. Install it with: pip install -U opendataloader-pdf
```

## Test coverage

Unit tests for this integration:

- `tests/test_pdf_ingestion.py`
- `tests/test_ingest_project_pdfs_script.py`

Run:

```powershell
python -m unittest tests/test_pdf_ingestion.py tests/test_ingest_project_pdfs_script.py
```
