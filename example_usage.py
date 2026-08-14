"""
example_usage.py — End-to-End PrivaRepo Demo

Demonstrates all major features of the PrivaRepo pipeline:
1. Index a sample Python project (uses this project's own source)
2. Query with hybrid retrieval + reranking + LLM
3. Raw search (no LLM)
4. Filtered search by language and chunk type
5. Statistics
6. Export / Import

Run this after installing dependencies and starting Ollama:
    $ ollama pull qwen2.5-coder:7b
    $ python example_usage.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging for the demo
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.resolve()


def separator(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def check_ollama(pipeline) -> bool:
    """Verify Ollama is reachable before running LLM-dependent demos."""
    if pipeline.llm.is_available():
        print("✅ Ollama is available")
        return True
    else:
        print("⚠  Ollama not available — skipping LLM demos.")
        print(f"   Start Ollama and run: ollama pull {pipeline.config.llm.model}")
        return False


def demo_index(pipeline) -> None:
    separator("1. INDEXING — Parsing this repository's source code")
    print(f"Indexing: {PROJECT_ROOT}")

    result = pipeline.index_repository(
        repo_path=str(PROJECT_ROOT),
        show_progress=True,
    )

    print(f"\n📦 Indexing Results:")
    for key, val in result.items():
        print(f"   {key}: {val}")


def demo_search(pipeline) -> None:
    separator("2. HYBRID SEARCH — No LLM, pure retrieval")

    queries = [
        ("How does BM25 search work?", None, None),
        ("cross encoder reranking", None, "function"),
        ("class extraction visitor pattern", "python", None),
        ("metadata extraction file path", None, None),
    ]

    for query, language, chunk_type in queries:
        print(f"\n🔍 Query: '{query}'")
        if language:
            print(f"   Filter: language={language}")
        if chunk_type:
            print(f"   Filter: chunk_type={chunk_type}")

        t0 = time.monotonic()
        chunks, timings = pipeline.search(
            query=query,
            language=language,
            chunk_type=chunk_type,
        )
        elapsed = (time.monotonic() - t0) * 1000

        print(f"   Results: {len(chunks)} chunks in {elapsed:.1f}ms")
        for i, chunk in enumerate(chunks[:3], 1):
            name = chunk.function_name or chunk.class_name or "—"
            file_name = Path(chunk.file_path).name if chunk.file_path else "—"
            print(
                f"   [{i}] {chunk.chunk_type} '{name}' — {file_name} "
                f"L{chunk.start_line}–{chunk.end_line} "
                f"(rerank: {chunk.rerank_score:.3f})"
            )


def demo_query(pipeline, llm_available: bool) -> None:
    separator("3. RAG QUERY — Full retrieval + LLM generation")

    if not llm_available:
        print("   Skipped (Ollama not available)")
        return

    questions = [
        ("How does Reciprocal Rank Fusion work in this codebase?", "explain"),
        ("What are potential issues with the BM25 tokenization?", "find_bugs"),
        ("How is the cross encoder configured?", "general"),
    ]

    for question, task_type in questions:
        print(f"\n❓ Q: {question}")
        print(f"   Task type: {task_type}")

        t0 = time.monotonic()
        response = pipeline.query(question=question, task_type=task_type)
        elapsed = time.monotonic() - t0

        print(f"\n💬 Answer (truncated to 300 chars):")
        print(f"   {response.answer[:300]}...")
        print(f"\n   📁 Referenced files: {response.referenced_files[:3]}")
        print(f"   🔧 Functions used: {response.functions_used[:3]}")
        print(
            f"   ⏱  Retrieval: {response.retrieval_time*1000:.0f}ms | "
            f"Generation: {response.generation_time*1000:.0f}ms | "
            f"Total: {elapsed*1000:.0f}ms"
        )
        print()


def demo_stats(pipeline) -> None:
    separator("4. STATISTICS — Collection summary")

    stats = pipeline.get_stats()
    print(f"\n📊 Collection Statistics:")
    print(f"   Total chunks:     {stats.get('total_chunks', 0)}")
    print(f"   Unique files:     {stats.get('unique_files', 0)}")
    print(f"   Embedding model:  {stats.get('embedding_model', '—')}")
    print(f"   BM25 index size:  {stats.get('bm25_index_size', 0)}")
    print(f"\n   Language breakdown: {stats.get('languages', {})}")
    print(f"   Chunk type breakdown: {stats.get('chunk_types', {})}")


def demo_export_import(pipeline) -> None:
    separator("5. EXPORT / IMPORT — Collection portability")

    export_path = PROJECT_ROOT / "demo_export.ndjson"

    print(f"📤 Exporting to: {export_path}")
    n_exported = pipeline.export(str(export_path))
    print(f"   Exported {n_exported} records ({export_path.stat().st_size / 1024:.1f} KB)")

    # Verify import (into the same collection — effectively a no-op upsert)
    print(f"\n📥 Re-importing from: {export_path}")
    n_imported = pipeline.import_data(str(export_path), reset_first=False)
    print(f"   Imported {n_imported} records")

    # Cleanup
    export_path.unlink(missing_ok=True)
    print("   Export file cleaned up.")


def demo_filtered_search(pipeline) -> None:
    separator("6. FILTERED SEARCH — Language and type filtering")

    filter_tests = [
        {"query": "find all class definitions", "chunk_type": "class"},
        {"query": "async function handling", "language": "python", "chunk_type": "function"},
        {"query": "configuration dataclass", "language": "python"},
    ]

    for test in filter_tests:
        query = test["query"]
        lang = test.get("language")
        ctype = test.get("chunk_type")

        print(f"\n🔎 '{query}' [lang={lang}, type={ctype}]")
        chunks, _ = pipeline.search(query=query, language=lang, chunk_type=ctype)

        if chunks:
            for c in chunks[:2]:
                name = c.function_name or c.class_name or "—"
                print(f"   → {c.chunk_type} '{name}' in {Path(c.file_path).name}")
        else:
            print("   No results")


def main() -> None:
    print("=" * 70)
    print("  PrivaRepo — Fully Local AI Code Intelligence (Demo)")
    print("=" * 70)

    # Initialise pipeline
    from config import AppConfig
    from rag_pipeline import RAGPipeline

    print("\nInitialising pipeline...")
    cfg = AppConfig()
    pipeline = RAGPipeline(cfg)
    llm_available = check_ollama(pipeline)

    # Run demos
    demo_index(pipeline)
    demo_search(pipeline)
    demo_query(pipeline, llm_available)
    demo_stats(pipeline)
    demo_export_import(pipeline)
    demo_filtered_search(pipeline)

    separator("DEMO COMPLETE")
    print("\n✅ All demos finished successfully.")
    print("\nNext steps:")
    print("  • Index your own codebase:  python -m cli index /path/to/repo")
    print("  • Ask a question:           python -m cli query 'How does auth work?'")
    print("  • Interactive mode:         python -m cli interactive")
    print("  • View stats:               python -m cli stats")
    print("  • Run benchmarks:           python -m cli benchmark")
    print()


if __name__ == "__main__":
    main()
