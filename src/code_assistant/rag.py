from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
import certifi
import httpx
from pydantic import BaseModel, Field
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

QDRANT_LOCAL_LOCK = threading.RLock()
CorrectiveRAGMode = Literal["fast", "balanced", "aggressive"]


@dataclass(frozen=True)
class RetrievalBundle:
    context: str
    sources: list[dict[str, str]]
    chunks: int
    indexed_now: bool = False
    detail: str = ""
    retrieval_query: str = ""
    compression_ratio: float = 1.0


class CorrectiveRAGDecision(BaseModel):
    score: int = Field(ge=1, le=5)
    verdict: str = Field(description="Short retrieval quality verdict.")
    should_retry: bool = Field(description="Whether retrieval should be retried with a rewritten query.")
    rewritten_query: str = Field(default="", description="Improved retrieval query when retrying is helpful.")


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
        retrieval_fetch_k: int = 10,
        max_chunks_per_source: int = 2,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        auto_index: bool = False,
        corrective_enabled: bool = False,
        corrective_model: str = "mistral-small-latest",
        corrective_mode: CorrectiveRAGMode = "balanced",
        corrective_min_score: int = 3,
        corrective_retry_k: int = 6,
        include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.qdrant_path = Path(qdrant_path).resolve()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        self.retrieval_fetch_k = max(retrieval_fetch_k, retrieval_k)
        self.max_chunks_per_source = max(1, max_chunks_per_source)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_index = auto_index
        self.corrective_enabled = corrective_enabled
        self.corrective_model = corrective_model
        self.corrective_mode: CorrectiveRAGMode = (
            corrective_mode
            if corrective_mode in {"fast", "balanced", "aggressive"}
            else "balanced"
        )
        self.corrective_min_score = corrective_min_score
        self.corrective_retry_k = corrective_retry_k
        self.include_patterns = include_patterns
        self._corrective_chain = (
            self._build_corrective_chain()
            if corrective_enabled and self._mode_uses_grader()
            else None
        )

    def index_project(self, *, force: bool = True) -> dict[str, int]:
        documents, file_count = self._build_documents()
        if not documents:
            return {"files": 0, "chunks": 0}

        with QDRANT_LOCAL_LOCK:
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

    def index_attachment_texts(
        self,
        *,
        attachments: list[dict[str, str]],
    ) -> dict[str, int]:
        """Incrementally index uploaded attachment text into the same Qdrant collection."""
        documents: list[Document] = []
        file_count = 0
        for item in attachments:
            filename = str(item.get("filename", "attachment.txt") or "attachment.txt")
            content = str(item.get("content", "") or "").strip()
            attachment_id = str(item.get("attachment_id", "") or "")
            if not content:
                continue
            file_count += 1
            checksum = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
            pseudo_path = self.project_root / filename
            source = f"attachment:{attachment_id}:{filename}" if attachment_id else f"attachment:{filename}"
            for chunk_index, chunk in enumerate(self._split_text(pseudo_path, content)):
                if not chunk.strip():
                    continue
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "chunk_index": str(chunk_index),
                            "checksum": checksum,
                            "kind": "attachment",
                        },
                    )
                )
        if not documents:
            return {"files": 0, "chunks": 0}

        with QDRANT_LOCAL_LOCK:
            client = self._create_client()
            try:
                if not client.collection_exists(self.collection_name):
                    store = QdrantVectorStore.from_documents(
                        documents,
                        embedding=self._build_embeddings(),
                        path=str(self.qdrant_path),
                        collection_name=self.collection_name,
                    )
                    if hasattr(store, "client"):
                        store.client.close()
                else:
                    store = QdrantVectorStore(
                        client=client,
                        collection_name=self.collection_name,
                        embedding=self._build_embeddings(),
                    )
                    store.add_documents(documents)
            finally:
                client.close()
        return {"files": file_count, "chunks": len(documents)}

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

        adaptive_k = self._adaptive_retrieval_k(query)
        dense_error = ""
        try:
            dense_documents = self._similarity_search(query, k=max(self.retrieval_fetch_k, adaptive_k + 2))
        except Exception as exc:
            dense_documents = []
            dense_error = str(exc)
        keyword_documents = self._keyword_search(query, k=max(4, adaptive_k))
        documents = self._hybrid_merge(query, dense_documents, keyword_documents, limit_k=adaptive_k)

        unique_sources: list[str] = []
        seen_sources: set[str] = set()
        source_rows: list[dict[str, str]] = []
        context_parts: list[str] = []
        active_query = query
        detail = "RAG retrieval ran, but no relevant project chunks were found."

        if not documents and self.corrective_enabled:
            retry_query = self._fallback_rewrite(query)
            try:
                retried_dense = self._similarity_search(
                    retry_query,
                    k=self._effective_retry_k(),
                )
            except Exception:
                retried_dense = []
            retried_keyword = self._keyword_search(retry_query, k=max(4, adaptive_k))
            retried_documents = self._hybrid_merge(
                query,
                retried_dense,
                retried_keyword,
                limit_k=self.retrieval_k,
            )
            if retried_documents:
                documents = retried_documents
                active_query = retry_query
                detail = (
                    f"Corrective RAG ({self.corrective_mode}) retried retrieval "
                    "after an empty initial result."
                )

        if documents and self.corrective_enabled and self._corrective_chain is not None:
            decision = self._grade_retrieval(query, documents)
            if decision is not None and (
                decision.score < self._effective_min_score() or decision.should_retry
            ):
                retry_query = decision.rewritten_query.strip() or self._fallback_rewrite(query)
                try:
                    retried_dense = self._similarity_search(
                        retry_query,
                        k=self._effective_retry_k(),
                    )
                except Exception:
                    retried_dense = []
                retried_keyword = self._keyword_search(retry_query, k=max(4, adaptive_k))
                retried_documents = self._hybrid_merge(
                    query,
                    retried_dense,
                    retried_keyword,
                    limit_k=self.retrieval_k,
                )
                retried_decision = self._grade_retrieval(query, retried_documents)
                if retried_documents and self._should_use_retry(decision, retried_decision):
                    documents = retried_documents
                    active_query = retry_query
                    if retried_decision is not None:
                        detail = (
                            f"Corrective RAG ({self.corrective_mode}) retried retrieval with a rewritten query and "
                            f"accepted the result at score {retried_decision.score}/5."
                        )
                    else:
                        detail = (
                            f"Corrective RAG ({self.corrective_mode}) retried retrieval "
                            "with a rewritten query."
                        )
                else:
                    detail = (
                        f"Corrective RAG ({self.corrective_mode}) kept the original retrieval after grading it at "
                        f"score {decision.score}/5."
                    )

        for index, document in enumerate(documents, start=1):
            source = str(document.metadata.get("source", "unknown"))
            chunk_index = str(document.metadata.get("chunk_index", "0"))
            if source not in seen_sources:
                unique_sources.append(source)
                seen_sources.add(source)
            source_rows.append({"source": source, "chunk_index": chunk_index})
            compressed = self._compress_chunk(query, document.page_content.strip())
            context_parts.append(
                f"[{index}] Source: {source}\n"
                f"Chunk: {chunk_index}\n"
                f"{compressed}"
            )

        if not context_parts:
            return RetrievalBundle(
                context="",
                sources=[],
                chunks=0,
                indexed_now=indexed_now,
                detail=detail,
                retrieval_query=active_query,
            )

        if detail == "RAG retrieval ran, but no relevant project chunks were found.":
            detail_prefix = "Indexed project files and " if indexed_now else ""
            detail = f"{detail_prefix}retrieved {len(context_parts)} chunk(s) from {len(unique_sources)} file(s)."
            if dense_error:
                detail += " Dense retrieval unavailable; used keyword fallback."
        full_context = "\n\n".join(context_parts)
        raw_chars = sum(len(str(document.page_content)) for document in documents)
        compressed_chars = len(full_context)
        ratio = round(compressed_chars / max(1, raw_chars), 3)
        return RetrievalBundle(
            context=full_context,
            sources=source_rows,
            chunks=len(context_parts),
            indexed_now=indexed_now,
            detail=detail,
            retrieval_query=active_query,
            compression_ratio=ratio,
        )

    def _build_corrective_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are grading retrieval quality for a codebase-aware assistant. "
                        "Score the retrieved context from 1 to 5 against the user's original query. "
                        "If the retrieval is weak, produce a better search query focused on filenames, "
                        "symbols, endpoints, environment variables, and implementation keywords. "
                        "Prefer keeping the original retrieval when it is already strong."
                    ),
                ),
                (
                    "human",
                    (
                        "Original query:\n{query}\n\n"
                        "Retrieved context preview:\n{context_preview}\n\n"
                        "Return whether retrieval should be retried."
                    ),
                ),
            ]
        )
        llm = ChatMistralAI(
            model=self.corrective_model,
            temperature=0.0,
        )
        return prompt | llm.with_structured_output(CorrectiveRAGDecision)

    def _mode_uses_grader(self) -> bool:
        return self.corrective_mode in {"balanced", "aggressive"}

    def _effective_min_score(self) -> int:
        if self.corrective_mode == "aggressive":
            return max(self.corrective_min_score, 4)
        return self.corrective_min_score

    def _effective_retry_k(self) -> int:
        if self.corrective_mode == "aggressive":
            return max(self.corrective_retry_k, self.retrieval_fetch_k + 2)
        if self.corrective_mode == "fast":
            return self.retrieval_fetch_k
        return max(self.retrieval_fetch_k, self.corrective_retry_k)

    def _grade_retrieval(
        self,
        query: str,
        documents: list[Document],
    ) -> CorrectiveRAGDecision | None:
        if not documents or self._corrective_chain is None:
            return None
        try:
            context_preview = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content[:700].strip()}"
                for doc in documents[:3]
            )
            return self._corrective_chain.invoke(
                {"query": query, "context_preview": context_preview}
            )
        except Exception:
            return None

    @staticmethod
    def _should_use_retry(
        initial: CorrectiveRAGDecision | None,
        retried: CorrectiveRAGDecision | None,
    ) -> bool:
        if retried is None:
            return False
        if initial is None:
            return True
        return retried.score >= initial.score

    @staticmethod
    def _fallback_rewrite(query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return normalized
        return (
            f"{normalized}\n"
            "Focus on project files, Python implementation details, FastAPI routes, "
            "settings, request models, and runtime validation."
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower())
            if len(token) >= 3
        }

    def _source_boost(self, source: str) -> float:
        lowered = source.lower()
        if lowered.startswith("src/"):
            return 0.35
        if lowered.startswith("public/"):
            return 0.15
        if lowered.startswith("docs/"):
            return 0.1
        if lowered.endswith(".py"):
            return 0.2
        return 0.0

    def _doc_score(self, query_tokens: set[str], document: Document) -> float:
        source = str(document.metadata.get("source", "unknown"))
        content_tokens = self._tokenize(document.page_content[:1800])
        overlap = len(query_tokens & content_tokens)
        overlap_score = min(1.0, overlap / max(1, len(query_tokens)))
        length_penalty = 0.0 if len(document.page_content) >= 120 else -0.1
        return overlap_score + self._source_boost(source) + length_penalty

    def _rerank_documents(
        self,
        query: str,
        documents: list[Document],
        *,
        limit_k: int,
    ) -> list[Document]:
        if not documents:
            return []
        query_tokens = self._tokenize(query)
        scored = [
            (self._doc_score(query_tokens, document), index, document)
            for index, document in enumerate(documents)
        ]
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)

        deduped: list[Document] = []
        seen_signatures: set[tuple[str, str]] = set()
        per_source: dict[str, int] = {}
        for _, _, document in scored:
            source = str(document.metadata.get("source", "unknown"))
            normalized_content = " ".join(document.page_content.split())[:300]
            signature = (source, normalized_content)
            if signature in seen_signatures:
                continue
            if per_source.get(source, 0) >= self.max_chunks_per_source:
                continue
            seen_signatures.add(signature)
            per_source[source] = per_source.get(source, 0) + 1
            deduped.append(document)
            if len(deduped) >= limit_k:
                break
        return deduped

    def _adaptive_retrieval_k(self, query: str) -> int:
        query_tokens = self._tokenize(query)
        if len(query_tokens) <= 5:
            return min(self.retrieval_k + 2, self.retrieval_fetch_k)
        if len(query_tokens) >= 16:
            return max(3, self.retrieval_k - 1)
        return self.retrieval_k

    def _keyword_search(self, query: str, *, k: int) -> list[Document]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        docs: list[Document] = []
        for path in self._iter_project_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            chunks = self._split_text(path, text)
            source = path.relative_to(self.project_root).as_posix()
            for idx, chunk in enumerate(chunks):
                tokens = self._tokenize(chunk[:1800])
                overlap = len(query_tokens & tokens)
                if overlap == 0:
                    continue
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "chunk_index": str(idx),
                            "keyword_overlap": str(overlap),
                        },
                    )
                )
        docs.sort(
            key=lambda d: int(str(d.metadata.get("keyword_overlap", "0"))),
            reverse=True,
        )
        return docs[:k]

    def _hybrid_merge(
        self,
        query: str,
        dense_documents: list[Document],
        keyword_documents: list[Document],
        *,
        limit_k: int,
    ) -> list[Document]:
        dense_ranked = self._rerank_documents(query, dense_documents, limit_k=max(limit_k, self.retrieval_k))
        all_docs = dense_ranked + keyword_documents
        return self._rerank_documents(query, all_docs, limit_k=limit_k)

    def _compress_chunk(self, query: str, chunk: str) -> str:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) <= 12:
            return chunk
        query_tokens = self._tokenize(query)
        scored_lines = []
        for index, line in enumerate(lines):
            line_tokens = self._tokenize(line)
            overlap = len(query_tokens & line_tokens)
            weight = overlap + (0.25 if line.startswith(("def ", "class ", "import ", "from ")) else 0.0)
            scored_lines.append((weight, index, line))
        scored_lines.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        keep = sorted(scored_lines[: min(12, len(scored_lines))], key=lambda row: row[1])
        return "\n".join(line for _, _, line in keep)

    def _similarity_search(self, query: str, *, k: int) -> list[Document]:
        with QDRANT_LOCAL_LOCK:
            client = self._create_client()
            try:
                store = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                    embedding=self._build_embeddings(),
                )
                return store.similarity_search(query, k=k)
            finally:
                client.close()

    def _collection_exists(self) -> bool:
        with QDRANT_LOCAL_LOCK:
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
        insecure_ssl = os.getenv("CODE_ASSISTANT_INSECURE_SSL", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        verify = False if insecure_ssl else certifi.where()
        endpoint = "https://api.mistral.ai/v1/"
        client = httpx.Client(base_url=endpoint, verify=verify, timeout=120)
        async_client = httpx.AsyncClient(base_url=endpoint, verify=verify, timeout=120)
        return MistralAIEmbeddings(
            model=self.embedding_model,
            client=client,
            async_client=async_client,
        )

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
