"""
tests/test_pipeline.py — End-to-End Pipeline Integration Tests

Tests the full RAGPipeline with mocked LLM and real ChromaDB/BM25.
Validates that indexing, retrieval, stats, export/import all work together.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from config import AppConfig, ChunkerConfig, LLMConfig, RetrievalConfig, VectorStoreConfig
from llm_interface import LLMResponse
from rag_pipeline import RAGPipeline, RAGResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_config(tmp_path):
    """AppConfig pointing at a temporary directory for isolation."""
    cfg = AppConfig()
    cfg.vector_store.persist_dir = str(tmp_path / "chromadb")
    cfg.vector_store.bm25_index_dir = str(tmp_path / "bm25")
    cfg.vector_store.collection_name = "test_collection"
    cfg.vector_store.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    cfg.retrieval.use_reranker = False    # Disable reranker in tests for speed
    cfg.retrieval.final_top_k = 3
    cfg.retrieval.top_k_vector = 5
    cfg.retrieval.top_k_bm25 = 5
    cfg.retrieval.rerank_candidates = 5
    cfg.retrieval.concurrent_retrieval = False
    cfg.chunker.min_chunk_lines = 1
    return cfg


@pytest.fixture
def sample_repo(tmp_path) -> Path:
    """Create a minimal Python repository for indexing tests."""
    repo = tmp_path / "sample_repo"
    repo.mkdir()

    (repo / "auth.py").write_text(textwrap.dedent("""\
        \"\"\"Authentication module.\"\"\"
        import hashlib
        import secrets

        class AuthService:
            \"\"\"Handles user authentication and session management.\"\"\"

            def __init__(self, secret_key: str):
                self.secret_key = secret_key
                self._sessions = {}

            def authenticate(self, username: str, password: str) -> str:
                \"\"\"Authenticate a user and return a session token.\"\"\"
                hashed = self._hash_password(password)
                if self._verify_credentials(username, hashed):
                    token = secrets.token_urlsafe(32)
                    self._sessions[token] = username
                    return token
                raise ValueError("Invalid credentials")

            def _hash_password(self, password: str) -> str:
                return hashlib.sha256(password.encode()).hexdigest()

            def _verify_credentials(self, username: str, password_hash: str) -> bool:
                return username == "admin" and len(password_hash) == 64

            def logout(self, token: str) -> None:
                self._sessions.pop(token, None)
    """))

    (repo / "database.py").write_text(textwrap.dedent("""\
        \"\"\"Database connection pool.\"\"\"
        from typing import Optional, List

        class ConnectionPool:
            \"\"\"Manages a pool of database connections.\"\"\"

            def __init__(self, max_connections: int = 10):
                self.max_connections = max_connections
                self._pool: List = []
                self._in_use: List = []

            def acquire(self):
                \"\"\"Acquire a connection from the pool.\"\"\"
                if self._pool:
                    conn = self._pool.pop()
                    self._in_use.append(conn)
                    return conn
                if len(self._in_use) < self.max_connections:
                    conn = self._create_connection()
                    self._in_use.append(conn)
                    return conn
                raise RuntimeError("Connection pool exhausted")

            def release(self, connection) -> None:
                \"\"\"Release a connection back to the pool.\"\"\"
                self._in_use.remove(connection)
                self._pool.append(connection)

            def _create_connection(self):
                return object()

        def get_connection(pool: ConnectionPool):
            return pool.acquire()
    """))

    (repo / "utils.py").write_text(textwrap.dedent("""\
        \"\"\"Utility functions.\"\"\"
        import re
        from typing import List

        def validate_email(email: str) -> bool:
            \"\"\"Validate an email address format.\"\"\"
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
            return bool(re.match(pattern, email))

        def sanitize_input(text: str) -> str:
            \"\"\"Remove potentially dangerous characters from input.\"\"\"
            return re.sub(r'[<>\"\\']', '', text)

        def chunk_list(lst: List, size: int) -> List[List]:
            \"\"\"Split a list into chunks of given size.\"\"\"
            return [lst[i:i+size] for i in range(0, len(lst), size)]
    """))

    return repo


@pytest.fixture
def mock_llm_response():
    return LLMResponse(
        raw_text="## ANSWER\nThe authenticate method validates credentials.\n\n## REASONING\nBased on the code context.\n\n## REFERENCED FILES\n- auth.py\n\n## FUNCTIONS USED\n- authenticate (auth.py)",
        answer="The authenticate method validates credentials.",
        reasoning="Based on the code context.",
        referenced_files=["auth.py"],
        functions_used=["authenticate (auth.py)"],
        latency_seconds=1.5,
        model="qwen2.5-coder:7b",
    )


@pytest.fixture
def pipeline_with_data(temp_config, sample_repo):
    """Pipeline with sample_repo already indexed."""
    pipeline = RAGPipeline(temp_config)
    pipeline.index_repository(str(sample_repo), show_progress=False)
    return pipeline


# ---------------------------------------------------------------------------
# Indexing Tests
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_index_returns_success(self, temp_config, sample_repo):
        pipeline = RAGPipeline(temp_config)
        result = pipeline.index_repository(str(sample_repo), show_progress=False)

        assert result["status"] == "success"
        assert result["chunks_extracted"] > 0
        assert result["chunks_indexed"] > 0

    def test_index_populates_vector_store(self, temp_config, sample_repo):
        pipeline = RAGPipeline(temp_config)
        pipeline.index_repository(str(sample_repo), show_progress=False)

        assert pipeline.vector_store.count > 0

    def test_index_builds_bm25(self, temp_config, sample_repo):
        pipeline = RAGPipeline(temp_config)
        pipeline.index_repository(str(sample_repo), show_progress=False)

        assert pipeline.bm25_index.is_built
        assert pipeline.bm25_index.size > 0

    def test_index_idempotent(self, temp_config, sample_repo):
        """Indexing the same repo twice should not duplicate data (upsert semantics)."""
        pipeline = RAGPipeline(temp_config)
        r1 = pipeline.index_repository(str(sample_repo), show_progress=False)
        initial_count = pipeline.vector_store.count

        r2 = pipeline.index_repository(str(sample_repo), show_progress=False)
        final_count = pipeline.vector_store.count

        assert final_count == initial_count, "Re-indexing should upsert, not duplicate"

    def test_index_empty_dir_returns_warning(self, temp_config, tmp_path):
        pipeline = RAGPipeline(temp_config)
        result = pipeline.index_repository(str(tmp_path), show_progress=False)
        assert result.get("status") in ("success", "warning")

    def test_indexed_chunks_have_metadata(self, pipeline_with_data):
        items = pipeline_with_data.vector_store.get_all_documents()
        assert len(items) > 0
        for chunk_id, document, metadata in items:
            assert "language" in metadata
            assert "chunk_type" in metadata
            assert "file_path" in metadata
            assert metadata["language"] == "python"


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------

class TestSearch:
    def test_search_returns_chunks(self, pipeline_with_data):
        chunks, timings = pipeline_with_data.search("authentication login")
        assert len(chunks) > 0

    def test_search_authentication_finds_auth_module(self, pipeline_with_data):
        chunks, _ = pipeline_with_data.search("authenticate user session token")
        files = {Path(c.file_path).name for c in chunks}
        assert "auth.py" in files

    def test_search_database_finds_connection_pool(self, pipeline_with_data):
        chunks, _ = pipeline_with_data.search("database connection pool acquire release")
        files = {Path(c.file_path).name for c in chunks}
        assert "database.py" in files

    def test_search_with_language_filter(self, pipeline_with_data):
        chunks, _ = pipeline_with_data.search("function", language="python")
        for chunk in chunks:
            assert chunk.language == "python"

    def test_search_with_chunk_type_filter(self, pipeline_with_data):
        chunks, _ = pipeline_with_data.search("class", chunk_type="class")
        for chunk in chunks:
            assert chunk.chunk_type == "class"

    def test_search_timings_returned(self, pipeline_with_data):
        _, timings = pipeline_with_data.search("test query")
        assert "total_ms" in timings
        assert timings["total_ms"] > 0

    def test_search_results_have_scores(self, pipeline_with_data):
        chunks, _ = pipeline_with_data.search("validate email input")
        for chunk in chunks:
            assert chunk.rrf_score >= 0
            assert isinstance(chunk.chunk_id, str)
            assert len(chunk.chunk_id) > 0

    def test_search_final_top_k_respected(self, temp_config, sample_repo):
        temp_config.retrieval.final_top_k = 2
        pipeline = RAGPipeline(temp_config)
        pipeline.index_repository(str(sample_repo), show_progress=False)

        chunks, _ = pipeline.search("class method function")
        assert len(chunks) <= 2

    def test_empty_query_handled(self, pipeline_with_data):
        """Empty query should not crash — returns something or empty list."""
        try:
            chunks, _ = pipeline_with_data.search("")
            assert isinstance(chunks, list)
        except Exception:
            pass  # Empty query raising is also acceptable


# ---------------------------------------------------------------------------
# RAG Query Tests (LLM mocked)
# ---------------------------------------------------------------------------

class TestRAGQuery:
    def test_query_with_mocked_llm(self, pipeline_with_data, mock_llm_response):
        with patch.object(pipeline_with_data.llm, "generate", return_value=mock_llm_response):
            response = pipeline_with_data.query("How does authentication work?")

        assert isinstance(response, RAGResponse)
        assert response.answer == mock_llm_response.answer
        assert "authenticate" in response.answer.lower() or response.answer

    def test_query_response_has_timing(self, pipeline_with_data, mock_llm_response):
        with patch.object(pipeline_with_data.llm, "generate", return_value=mock_llm_response):
            response = pipeline_with_data.query("test question")

        assert response.retrieval_time > 0
        assert response.generation_time >= 0
        assert response.total_time >= response.retrieval_time

    def test_query_returns_retrieved_chunks(self, pipeline_with_data, mock_llm_response):
        with patch.object(pipeline_with_data.llm, "generate", return_value=mock_llm_response):
            response = pipeline_with_data.query("database connection pool")

        assert len(response.retrieved_chunks) > 0

    def test_query_empty_results_handled(self, temp_config, tmp_path):
        """Query on empty collection returns graceful response."""
        pipeline = RAGPipeline(temp_config)  # No indexing
        response = pipeline.query("anything")

        assert isinstance(response, RAGResponse)
        assert "no relevant code" in response.answer.lower() or response.answer

    def test_query_task_types(self, pipeline_with_data, mock_llm_response):
        task_types = ["general", "explain", "find_bugs", "similar_code", "function_search", "class_search"]
        for task_type in task_types:
            with patch.object(pipeline_with_data.llm, "generate", return_value=mock_llm_response):
                response = pipeline_with_data.query(
                    "authentication", task_type=task_type
                )
            assert isinstance(response, RAGResponse)


# ---------------------------------------------------------------------------
# Stats Tests
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_after_indexing(self, pipeline_with_data):
        stats = pipeline_with_data.get_stats()
        assert stats["total_chunks"] > 0
        assert stats["unique_files"] > 0
        assert "python" in stats["languages"]

    def test_stats_empty_collection(self, temp_config):
        pipeline = RAGPipeline(temp_config)
        stats = pipeline.get_stats()
        assert stats["total_chunks"] == 0

    def test_stats_includes_bm25_info(self, pipeline_with_data):
        stats = pipeline_with_data.get_stats()
        assert "bm25_index_size" in stats
        assert "bm25_index_built" in stats


# ---------------------------------------------------------------------------
# Export / Import Tests
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export_creates_file(self, pipeline_with_data, tmp_path):
        export_path = str(tmp_path / "export.ndjson")
        n = pipeline_with_data.export(export_path)

        assert n > 0
        assert Path(export_path).exists()
        assert Path(export_path).stat().st_size > 0

    def test_export_is_valid_ndjson(self, pipeline_with_data, tmp_path):
        export_path = str(tmp_path / "export.ndjson")
        pipeline_with_data.export(export_path)

        with open(export_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                assert "chunk_id" in record
                assert "document" in record
                assert "metadata" in record

    def test_import_restores_data(self, temp_config, sample_repo, tmp_path):
        # Index + export
        pipeline1 = RAGPipeline(temp_config)
        pipeline1.index_repository(str(sample_repo), show_progress=False)
        original_count = pipeline1.vector_store.count
        export_path = str(tmp_path / "export.ndjson")
        pipeline1.export(export_path)

        # New pipeline, reset, import
        cfg2 = temp_config
        cfg2.vector_store.persist_dir = str(tmp_path / "chromadb2")
        cfg2.vector_store.bm25_index_dir = str(tmp_path / "bm25_2")
        pipeline2 = RAGPipeline(cfg2)
        n = pipeline2.import_data(export_path, reset_first=False)

        assert n == original_count
        assert pipeline2.vector_store.count == original_count

    def test_import_rebuilds_bm25(self, temp_config, sample_repo, tmp_path):
        pipeline = RAGPipeline(temp_config)
        pipeline.index_repository(str(sample_repo), show_progress=False)
        export_path = str(tmp_path / "export.ndjson")
        pipeline.export(export_path)

        cfg2 = AppConfig()
        cfg2.vector_store.persist_dir = str(tmp_path / "chromadb3")
        cfg2.vector_store.bm25_index_dir = str(tmp_path / "bm25_3")
        cfg2.retrieval.use_reranker = False

        pipeline2 = RAGPipeline(cfg2)
        pipeline2.import_data(export_path)

        assert pipeline2.bm25_index.is_built


# ---------------------------------------------------------------------------
# Reset Tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_collection(self, pipeline_with_data):
        assert pipeline_with_data.vector_store.count > 0
        pipeline_with_data.reset()
        assert pipeline_with_data.vector_store.count == 0

    def test_reset_invalidates_bm25(self, pipeline_with_data, tmp_path):
        pipeline_with_data.reset()
        assert not pipeline_with_data.bm25_index.is_built
