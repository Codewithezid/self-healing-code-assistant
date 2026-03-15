from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

DEFAULT_INCLUDE_PATTERNS = (
    "src/**/*.py",
    "docs/**/*.md",
    "public/**/*.js",
    "public/**/*.html",
    "public/**/*.css",
    "README.md",
    "requirements.txt",
    "requirements-local.txt",
    ".env.example",
    "main.py",
    "web_main.py",
)

DEFAULT_EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archive",
    "artifacts",
    "data/runtime",
}


@dataclass(frozen=True)
class RetrievalBundle:
    context: str
    sources: list[dict[str, str]]
    chunks: int
    indexed_now: bool = False
    detail: str = ""


class ProjectRAG:
    """Local project RAG backed by a persistent Qdrant collection."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        qdrant_path: str | Path,
        collection_name: str,
        embedding_model: str = "mistral-embed",
        retrieval_k: int = 4,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        auto_index: bool = False,
        include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.qdrant_path = Path(qdrant_path).resolve()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_index = auto_index
        self.include_patterns = include_patterns

    def index_project(self, *, force: bool = True) -> dict[str, int]:
        documents, file_count = self._build_documents()
        if not documents:
            return {"files": 0, "chunks": 0}

        if force:
            client = self._create_client()
            try:
                if client.collection_exists(self.collection_name):
                    client.delete_collection(self.collection_name)
            finally:
                client.close()

        store = QdrantVectorStore.from_documents(
            documents,
            embedding=self._build_embeddings(),
            path=str(self.qdrant_path),
            collection_name=self.collection_name,
        )
        try:
            return {"files": file_count, "chunks": len(documents)}
        finally:
            if hasattr(store, "client"):
                store.client.close()

    def retrieve(self, query: str) -> RetrievalBundle:
        if not query.strip():
            return RetrievalBundle(
                context="",
                sources=[],
                chunks=0,
                detail="Skipped retrieval because the query was empty.",
            )

        indexed_now = False
        if not self._collection_exists():
            if not self.auto_index:
                return RetrievalBundle(
                    context="",
                    sources=[],
                    chunks=0,
                    detail=(
                        "RAG is enabled, but the local Qdrant index does not exist yet. "
                        "Run scripts/index_project_rag.py or enable auto-indexing."
                    ),
                )
            stats = self.index_project(force=True)
            indexed_now = True
            if stats["chunks"] == 0:
                return RetrievalBundle(
                    context="",
                    sources=[],
                    chunks=0,
                    indexed_now=True,
                    detail="RAG indexing ran, but no eligible project files were found.",
                )

        client = self._create_client()
        try:
            store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self._build_embeddings(),
            )
            documents = store.similarity_search(query, k=self.retrieval_k)
        finally:
            client.close()

        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        source_rows: list[dict[str, str]] = []
        context_parts: list[str] = []

        for index, document in enumerate(documents, start=1):
            source = str(document.metadata.get("source", "unknown"))
            chunk_index = str(document.metadata.get("chunk_index", "0"))
            if source not in seen_sources:
                unique_sources.append(source)
                seen_sources.add(source)
            source_rows.append({"source": source, "chunk_index": chunk_index})
            context_parts.append(
                f"[{index}] Source: {source}\n"
                f"Chunk: {chunk_index}\n"
                f"{document.page_content.strip()}"
            )

        if not context_parts:
            return RetrievalBundle(
                context="",
                sources=[],
                chunks=0,
                indexed_now=indexed_now,
                detail="RAG retrieval ran, but no relevant project chunks were found.",
            )

        detail_prefix = "Indexed project files and " if indexed_now else ""
        return RetrievalBundle(
            context="\n\n".join(context_parts),
            sources=source_rows,
            chunks=len(context_parts),
            indexed_now=indexed_now,
            detail=(
                f"{detail_prefix}retrieved {len(context_parts)} chunk(s) "
                f"from {len(unique_sources)} file(s)."
            ),
        )

    def _collection_exists(self) -> bool:
        client = self._create_client()
        try:
            return client.collection_exists(self.collection_name)
        finally:
            client.close()

    def _create_client(self) -> QdrantClient:
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(
            path=str(self.qdrant_path),
            force_disable_check_same_thread=True,
        )

    def _build_embeddings(self) -> MistralAIEmbeddings:
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "RAG embeddings require MISTRAL_API_KEY to be set."
            )
        return MistralAIEmbeddings(model=self.embedding_model)

    def _build_documents(self) -> tuple[list[Document], int]:
        documents: list[Document] = []
        file_count = 0

        for path in self._iter_project_files():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            file_count += 1
            relative_path = path.relative_to(self.project_root).as_posix()
            checksum = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            for chunk_index, chunk in enumerate(self._split_text(path, text)):
                if not chunk.strip():
                    continue
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": relative_path,
                            "chunk_index": str(chunk_index),
                            "checksum": checksum,
                        },
                    )
                )
        return documents, file_count

    def _iter_project_files(self) -> Iterable[Path]:
        seen: set[Path] = set()
        for pattern in self.include_patterns:
            for path in self.project_root.glob(pattern):
                resolved = path.resolve()
                if not resolved.is_file():
                    continue
                if resolved in seen:
                    continue
                if self._is_excluded(resolved):
                    continue
                seen.add(resolved)
                yield resolved

    def _is_excluded(self, path: Path) -> bool:
        relative = path.relative_to(self.project_root).as_posix()
        parts = set(path.relative_to(self.project_root).parts)
        if parts & DEFAULT_EXCLUDED_PARTS:
            return True
        return any(relative.startswith(prefix) for prefix in DEFAULT_EXCLUDED_PARTS)

    def _split_text(self, path: Path, text: str) -> list[str]:
        if path.suffix == ".py":
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return splitter.split_text(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)
