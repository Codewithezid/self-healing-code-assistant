from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.code_assistant.rag import ProjectRAG


class RagRerankTests(unittest.TestCase):
    def test_rerank_limits_chunks_per_source(self) -> None:
        rag = ProjectRAG(
            project_root=Path("."),
            qdrant_path=Path("data/qdrant-test"),
            collection_name="test-collection",
            retrieval_k=3,
            retrieval_fetch_k=8,
            max_chunks_per_source=1,
            auto_index=False,
        )
        docs = [
            Document(page_content="fastapi route user profile endpoint", metadata={"source": "src/api.py", "chunk_index": "0"}),
            Document(page_content="fastapi route user profile endpoint extended", metadata={"source": "src/api.py", "chunk_index": "1"}),
            Document(page_content="settings env var timeout retries", metadata={"source": "src/settings.py", "chunk_index": "0"}),
            Document(page_content="frontend css styles", metadata={"source": "public/styles.css", "chunk_index": "0"}),
        ]
        ranked = rag._rerank_documents("fastapi user endpoint settings", docs, limit_k=3)
        self.assertLessEqual(len(ranked), 3)
        unique_sources = {d.metadata.get("source") for d in ranked}
        self.assertEqual(len(unique_sources), len(ranked))


if __name__ == "__main__":
    unittest.main()
