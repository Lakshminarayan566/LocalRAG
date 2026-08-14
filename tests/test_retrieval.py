"""
tests/test_retrieval.py — Retrieval Pipeline Tests

Tests BM25 index, vector store, RRF fusion, and hybrid retrieval.
Uses in-memory or temporary ChromaDB instances where possible.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from config import RetrievalConfig
from rag_pipeline import (
    BM25Index,
    CrossEncoderReranker,
    HybridRetriever,
    RetrievedChunk,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(chunk_id: str, score: float, rank: int, document: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "document": document or f"Document for {chunk_id}",
        "metadata": {"language": "python", "chunk_type": "function", "function_name": chunk_id},
        "score": score,
        "rank": rank,
    }


def make_corpus(n: int = 20) -> list:
    """Generate a synthetic corpus for BM25 testing."""
    items = []
    for i in range(n):
        items.append((
            f"chunk_{i:04d}",
            f"function retrieve_items index BM25 vector search query result rank {i} "
            f"{'authentication login session' if i % 3 == 0 else ''} "
            f"{'database connection pool' if i % 4 == 0 else ''} "
            f"{'error handling exception retry' if i % 5 == 0 else ''}",
            {"language": "python", "chunk_type": "function", "function_name": f"func_{i}",
             "file_path": f"/src/module_{i % 5}.py"},
        ))
    return items


# ---------------------------------------------------------------------------
# BM25 Index Tests
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_build_and_search(self, tmp_path):
        index = BM25Index(str(tmp_path))
        corpus = make_corpus(20)
        index.build(corpus, save=True)

        assert index.is_built
        assert index.size == 20

        results = index.search("authentication login session", k=5)
        assert len(results) <= 5
        # All results should have chunk_ids from the corpus
        corpus_ids = {item[0] for item in corpus}
        for r in results:
            assert r["chunk_id"] in corpus_ids

    def test_search_returns_scores(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build(make_corpus(10))

        results = index.search("database connection", k=3)
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0 + 1e-6  # normalised

    def test_search_rank_order(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build(make_corpus(20))

        results = index.search("authentication login", k=10)
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1)), "Results not in rank order"

    def test_save_and_load(self, tmp_path):
        index = BM25Index(str(tmp_path))
        corpus = make_corpus(15)
        index.build(corpus, save=True)

        # Load into a fresh instance
        index2 = BM25Index(str(tmp_path))
        assert index2.load()
        assert index2.size == 15
        assert index2.is_built

        results = index2.search("retrieve items", k=3)
        assert len(results) > 0

    def test_empty_corpus(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build([], save=False)
        assert not index.is_built
        results = index.search("any query", k=5)
        assert results == []

    def test_language_filter(self, tmp_path):
        corpus = [
            ("py_chunk_0", "python function authentication", {"language": "python", "chunk_type": "function", "function_name": "auth"}),
            ("java_chunk_0", "java method authentication login", {"language": "java", "chunk_type": "method", "function_name": "auth"}),
            ("py_chunk_1", "python class authentication", {"language": "python", "chunk_type": "class", "function_name": "AuthService"}),
        ]
        index = BM25Index(str(tmp_path))
        index.build(corpus)

        python_results = index.search("authentication", k=5, language_filter="python")
        for r in python_results:
            assert r["metadata"]["language"] == "python"

    def test_tokenization_camelcase(self, tmp_path):
        """CamelCase tokens should be split for better matching."""
        corpus = [
            ("c1", "getUserAuthentication function returns User object", {"language": "python", "chunk_type": "function", "function_name": "f"}),
        ]
        index = BM25Index(str(tmp_path))
        index.build(corpus)

        # Should match on "user", "authentication" even from camelCase split
        results = index.search("user authentication", k=1)
        assert len(results) == 1

    def test_search_top_k_respected(self, tmp_path):
        index = BM25Index(str(tmp_path))
        index.build(make_corpus(50))

        for k in [1, 5, 10, 20]:
            results = index.search("query", k=k)
            assert len(results) <= k


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion Tests
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_basic_fusion(self):
        vector_results = [
            make_result("a", 0.95, 1),
            make_result("b", 0.90, 2),
            make_result("c", 0.80, 3),
        ]
        bm25_results = [
            make_result("b", 0.85, 1),
            make_result("d", 0.70, 2),
            make_result("a", 0.65, 3),
        ]
        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)

        # Both a and b appear in both lists, so should rank higher than c/d
        assert len(fused) == 4
        ids_in_order = [r["chunk_id"] for r in fused]
        # a and b should be in top 2
        assert set(ids_in_order[:2]) == {"a", "b"}

    def test_rrf_score_formula(self):
        """Verify RRF score = 1/(k+rank) for each list."""
        k = 60
        vector_results = [make_result("x", 1.0, 1)]
        bm25_results = [make_result("x", 1.0, 1)]

        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=k)
        expected_score = 2 * (1.0 / (k + 1))  # appears at rank 1 in both lists
        assert abs(fused[0]["rrf_score"] - expected_score) < 1e-9

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [], k=60)
        assert fused == []

    def test_single_list(self):
        results = [make_result("a", 0.9, 1), make_result("b", 0.8, 2)]
        fused = reciprocal_rank_fusion(results, k=60)
        assert len(fused) == 2

    def test_three_lists(self):
        """RRF should work with more than 2 result lists."""
        r1 = [make_result("a", 0.9, 1), make_result("b", 0.8, 2)]
        r2 = [make_result("a", 0.85, 1), make_result("c", 0.75, 2)]
        r3 = [make_result("a", 0.8, 1), make_result("d", 0.7, 2)]
        fused = reciprocal_rank_fusion(r1, r2, r3, k=60)

        # "a" appears in all 3 lists at rank 1 — must be top result
        assert fused[0]["chunk_id"] == "a"

    def test_rrf_scores_are_descending(self):
        r1 = [make_result(f"doc_{i}", 1.0 / (i + 1), i + 1) for i in range(10)]
        r2 = [make_result(f"doc_{i}", 1.0 / (i + 2), i + 1) for i in range(10)]
        fused = reciprocal_rank_fusion(r1, r2)
        scores = [r["rrf_score"] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_disjoint_results_all_included(self):
        r1 = [make_result("a", 1.0, 1)]
        r2 = [make_result("b", 1.0, 1)]
        fused = reciprocal_rank_fusion(r1, r2)
        ids = {r["chunk_id"] for r in fused}
        assert "a" in ids
        assert "b" in ids


# ---------------------------------------------------------------------------
# HybridRetriever Tests (with mocked vector store and BM25)
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_mock_vector_store(self, results: list):
        mock = MagicMock()
        mock.vector_search.return_value = results
        mock.metadata_filter.return_value = None
        return mock

    def _make_mock_bm25(self, results: list):
        mock = MagicMock()
        mock.search.return_value = results
        mock.is_built = True
        return mock

    def test_retrieve_returns_retrieved_chunks(self):
        vector_results = [make_result("a", 0.9, 1), make_result("b", 0.8, 2)]
        bm25_results = [make_result("b", 0.85, 1), make_result("c", 0.7, 2)]

        config = RetrievalConfig(
            top_k_vector=5,
            top_k_bm25=5,
            rerank_candidates=5,
            final_top_k=3,
            use_reranker=False,
            concurrent_retrieval=False,
        )
        retriever = HybridRetriever(
            vector_store=self._make_mock_vector_store(vector_results),
            bm25_index=self._make_mock_bm25(bm25_results),
            config=config,
        )

        chunks, timings = retriever.retrieve("test query")

        assert len(chunks) <= 3
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_rrf_scores_populated(self):
        vector_results = [make_result("a", 0.9, 1), make_result("b", 0.8, 2)]
        bm25_results = [make_result("a", 0.9, 1), make_result("c", 0.7, 2)]

        config = RetrievalConfig(
            top_k_vector=5,
            top_k_bm25=5,
            rerank_candidates=5,
            final_top_k=3,
            use_reranker=False,
            concurrent_retrieval=False,
        )
        retriever = HybridRetriever(
            vector_store=self._make_mock_vector_store(vector_results),
            bm25_index=self._make_mock_bm25(bm25_results),
            config=config,
        )

        chunks, _ = retriever.retrieve("test query")
        assert all(c.rrf_score > 0 for c in chunks)

    def test_timing_dict_returned(self):
        config = RetrievalConfig(
            use_reranker=False,
            concurrent_retrieval=False,
            final_top_k=2,
        )
        retriever = HybridRetriever(
            vector_store=self._make_mock_vector_store([make_result("a", 0.9, 1)]),
            bm25_index=self._make_mock_bm25([make_result("a", 0.8, 1)]),
            config=config,
        )
        _, timings = retriever.retrieve("query")

        assert "total_ms" in timings
        assert timings["total_ms"] >= 0

    def test_no_results_returns_empty(self):
        config = RetrievalConfig(use_reranker=False, concurrent_retrieval=False)
        retriever = HybridRetriever(
            vector_store=self._make_mock_vector_store([]),
            bm25_index=self._make_mock_bm25([]),
            config=config,
        )
        chunks, _ = retriever.retrieve("query")
        assert chunks == []

    def test_vector_scores_annotated(self):
        """RetrievedChunk should carry the vector score from the vector search."""
        v_results = [make_result("a", 0.95, 1), make_result("b", 0.80, 2)]
        b_results = [make_result("a", 0.70, 1)]

        config = RetrievalConfig(use_reranker=False, concurrent_retrieval=False, final_top_k=2)
        retriever = HybridRetriever(
            vector_store=self._make_mock_vector_store(v_results),
            bm25_index=self._make_mock_bm25(b_results),
            config=config,
        )
        chunks, _ = retriever.retrieve("query")

        chunk_a = next(c for c in chunks if c.chunk_id == "a")
        assert chunk_a.vector_score == pytest.approx(0.95)
        assert chunk_a.bm25_score == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# RetrievedChunk Tests
# ---------------------------------------------------------------------------

class TestRetrievedChunk:
    def test_raw_code_extraction(self):
        doc = "[FUNCTION] [python]\nName: foo\nFile: /src/foo.py\n\ndef foo():\n    return 42"
        chunk = RetrievedChunk(
            chunk_id="abc",
            document=doc,
            metadata={
                "chunk_type": "function",
                "language": "python",
                "file_path": "/src/foo.py",
                "function_name": "foo",
                "class_name": "",
                "start_line": 1,
                "end_line": 2,
            },
        )
        raw = chunk.raw_code
        assert "def foo():" in raw
        assert "return 42" in raw

    def test_properties(self):
        chunk = RetrievedChunk(
            chunk_id="xyz",
            document="code",
            metadata={
                "file_path": "/path/to/file.py",
                "function_name": "my_func",
                "class_name": "MyClass",
                "language": "python",
                "chunk_type": "method",
                "start_line": 10,
                "end_line": 20,
            },
        )
        assert chunk.file_path == "/path/to/file.py"
        assert chunk.function_name == "my_func"
        assert chunk.class_name == "MyClass"
        assert chunk.language == "python"
        assert chunk.chunk_type == "method"
        assert chunk.start_line == 10
        assert chunk.end_line == 20
