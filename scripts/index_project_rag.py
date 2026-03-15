from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.code_assistant.rag import ProjectRAG


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index local project files into a persistent Qdrant collection."
    )
    parser.add_argument(
        "--project-root",
        default=PROJECT_ROOT,
        help="Project root to index.",
    )
    parser.add_argument(
        "--qdrant-path",
        default="data/qdrant",
        help="Directory where the local Qdrant index will be stored.",
    )
    parser.add_argument(
        "--collection",
        default="code-assistant-project",
        help="Collection name to write into.",
    )
    parser.add_argument(
        "--embedding-model",
        default="mistral-embed",
        help="Embedding model name for MistralAIEmbeddings.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=4,
        help="Default retrieval fan-out stored with the config.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size used during indexing.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap used during indexing.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    load_dotenv(Path(args.project_root) / ".env")

    rag = ProjectRAG(
        project_root=args.project_root,
        qdrant_path=args.qdrant_path,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        retrieval_k=args.retrieval_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        auto_index=False,
    )
    stats = rag.index_project(force=True)
    print(
        "Indexed "
        f"{stats['files']} file(s) into {stats['chunks']} chunk(s) "
        f"at {Path(args.qdrant_path).resolve()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
