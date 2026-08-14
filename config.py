"""
config.py — PrivaRepo Configuration System

Centralised, strongly-typed configuration using Python dataclasses.
All defaults are tuned for production-quality retrieval quality.
Override any value via environment variables or a .env file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
_DEFAULT_PERSIST_DIR = _PROJECT_ROOT / ".chromadb"
_DEFAULT_BM25_INDEX_DIR = _PROJECT_ROOT / ".bm25"
_DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "eval_results"


# ---------------------------------------------------------------------------
# Chunker Configuration
# ---------------------------------------------------------------------------
@dataclass
class ChunkerConfig:
    """Controls Tree-sitter AST-boundary chunking behaviour."""

    # Languages to parse (file-extension → tree-sitter language name)
    supported_extensions: dict = field(
        default_factory=lambda: {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
        }
    )

    # Chunk types to extract per language
    chunk_types: List[str] = field(
        default_factory=lambda: ["function", "method", "class", "imports", "module"]
    )

    # Minimum source lines a chunk must have to be indexed
    # (filters out trivial one-liner stubs)
    min_chunk_lines: int = int(os.getenv("CHUNKER_MIN_LINES", "2"))

    # Maximum lines allowed in a single chunk before it is split at
    # logical sub-boundaries (nested functions / inner classes).
    max_chunk_lines: int = int(os.getenv("CHUNKER_MAX_LINES", "200"))

    # Lines of surrounding context to prepend to function/method chunks
    # (includes decorators, leading comments, class signature)
    context_lines: int = int(os.getenv("CHUNKER_CONTEXT_LINES", "3"))

    # Number of parallel workers for repository-level indexing
    num_workers: int = int(os.getenv("CHUNKER_WORKERS", "4"))

    # Glob patterns to exclude during repository walk
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/dist/**",
            "**/build/**",
            "**/*.min.js",
            "**/*.pyc",
        ]
    )


# ---------------------------------------------------------------------------
# Vector Store Configuration
# ---------------------------------------------------------------------------
@dataclass
class VectorStoreConfig:
    """ChromaDB and embedding model configuration."""

    # Where ChromaDB persists data on disk
    persist_dir: str = str(os.getenv("CHROMA_PERSIST_DIR", str(_DEFAULT_PERSIST_DIR)))

    # ChromaDB collection name
    collection_name: str = str(os.getenv("CHROMA_COLLECTION", "privarepo_code"))

    # HuggingFace sentence-transformer model for embedding
    # nomic-ai/nomic-embed-code is code-specialised and recommended.
    # Fall back to all-MiniLM-L6-v2 for faster indexing on CPU.
    embedding_model: str = str(
        os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # Batch size when calling embed() — balance memory vs. speed
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

    # ChromaDB upsert batch size
    upsert_batch_size: int = int(os.getenv("UPSERT_BATCH_SIZE", "256"))

    # Distance metric used by ChromaDB ("cosine" or "l2")
    distance_metric: str = "cosine"

    # BM25 serialised index directory
    bm25_index_dir: str = str(os.getenv("BM25_INDEX_DIR", str(_DEFAULT_BM25_INDEX_DIR)))


# ---------------------------------------------------------------------------
# Retrieval Configuration
# ---------------------------------------------------------------------------
@dataclass
class RetrievalConfig:
    """Hybrid retrieval and reranking configuration."""

    # Top-K candidates from vector search
    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "30"))

    # Top-K candidates from BM25
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "30"))

    # RRF constant (higher → less aggressive rank compression)
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Number of fused candidates to pass to the cross encoder
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))

    # Final top-K returned to the prompt builder
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))

    # Whether to apply cross encoder reranking (disable on very low-RAM machines)
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

    # HuggingFace cross encoder model
    cross_encoder_model: str = str(
        os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Device for cross encoder inference ("cpu", "cuda", "mps")
    reranker_device: str = str(os.getenv("RERANKER_DEVICE", "cpu"))

    # Minimum rerank score threshold — chunks below this are dropped
    rerank_score_threshold: float = float(os.getenv("RERANK_THRESHOLD", "-10.0"))

    # Concurrent retrieval: run vector + BM25 in parallel threads
    concurrent_retrieval: bool = True


# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Ollama LLM configuration."""

    # Ollama REST API base URL
    base_url: str = str(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # Model name served by Ollama
    model: str = str(os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"))
    # Sampling temperature (0 = deterministic, best for code Q&A)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Max tokens to generate. 2048 is overkill for a 4-section structured
    # answer (ANSWER/REASONING/REFERENCED FILES/FUNCTIONS USED) and is a
    # major contributor to 20s+ generation times on CPU/Metal. Lowered
    # default to 768 — raise via LLM_MAX_TOKENS if answers get truncated.
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "768"))

    # Request timeout in seconds
    timeout: int = int(os.getenv("LLM_TIMEOUT", "120"))

    # Number of retries on transient failures
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))

    # Retry base delay in seconds (exponential backoff)
    retry_base_delay: float = float(os.getenv("LLM_RETRY_DELAY", "1.0"))

    # Context window token limit for the model. Lowered from 8192 — 5
    # chunks x <=200 lines rarely approaches that, and a smaller KV cache
    # allocation speeds up both model load and prefill. Raise via
    # LLM_CONTEXT_WINDOW if you increase final_top_k or max_chunk_lines.
    context_window: int = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))

    # How long Ollama keeps the model resident in memory after a request.
    # Without this, Ollama's default (5m idle) can unload a 7B model
    # between calls, and reloading it costs several seconds — showing up
    # as "generation latency" that's actually model-load latency.
    keep_alive: str = str(os.getenv("LLM_KEEP_ALIVE", "30m"))


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Evaluation and benchmarking configuration."""

    # Path to JSON file containing evaluation queries + ground truth
    eval_queries_path: str = str(
        os.getenv(
            "EVAL_QUERIES_PATH",
            str(_PROJECT_ROOT / "data" / "eval_queries.json"),
        )
    )

    # Directory where evaluation result JSONs are saved
    results_dir: str = str(os.getenv("EVAL_RESULTS_DIR", str(_DEFAULT_RESULTS_DIR)))

    # Number of benchmark iterations for latency measurement
    benchmark_iterations: int = int(os.getenv("BENCHMARK_ITERS", "20"))

    # K for Precision@K and Recall@K
    eval_k: int = int(os.getenv("EVAL_K", "5"))

    # Faithfulness judge: "llm" (uses Ollama as judge) or "nli"
    faithfulness_method: str = str(os.getenv("FAITHFULNESS_METHOD", "llm"))

    # Metrics to compute (all enabled by default)
    compute_retrieval_metrics: bool = True
    compute_generation_metrics: bool = True
    compute_latency_metrics: bool = True


# ---------------------------------------------------------------------------
# Flask / API Configuration (optional)
# ---------------------------------------------------------------------------
@dataclass
class APIConfig:
    """Flask REST API configuration (optional serve mode)."""

    host: str = str(os.getenv("API_HOST", "127.0.0.1"))
    port: int = int(os.getenv("API_PORT", "8080"))
    debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Root Application Configuration
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    """
    Root configuration object.  Pass this single object throughout the app
    instead of individual sub-configs to keep dependency injection clean.
    """

    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Global log level
    log_level: str = str(os.getenv("LOG_LEVEL", "INFO"))

    def validate(self) -> None:
        """Validate configuration for common misconfigurations."""
        if self.retrieval.final_top_k > self.retrieval.rerank_candidates:
            raise ValueError(
                f"final_top_k ({self.retrieval.final_top_k}) must be <= "
                f"rerank_candidates ({self.retrieval.rerank_candidates})"
            )
        if self.retrieval.rerank_candidates > (
            self.retrieval.top_k_vector + self.retrieval.top_k_bm25
        ):
            raise ValueError(
                "rerank_candidates cannot exceed total vector+BM25 candidates"
            )
        if self.retrieval.final_top_k < self.eval.eval_k:
            logger.warning(
                "final_top_k (%d) < eval_k (%d) — retrieval metrics may be capped. "
                "Consider setting FINAL_TOP_K >= EVAL_K for accurate evaluation.",
                self.retrieval.final_top_k, self.eval.eval_k,
            )

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Factory: build AppConfig reading from environment / .env file."""
        cfg = cls()
        cfg.validate()
        return cfg

    def summary(self) -> dict:
        """Return a serialisable dict for logging / display purposes."""
        return {
            "embedding_model": self.vector_store.embedding_model,
            "collection": self.vector_store.collection_name,
            "persist_dir": self.vector_store.persist_dir,
            "llm_model": self.llm.model,
            "top_k_vector": self.retrieval.top_k_vector,
            "top_k_bm25": self.retrieval.top_k_bm25,
            "rrf_k": self.retrieval.rrf_k,
            "rerank_candidates": self.retrieval.rerank_candidates,
            "final_top_k": self.retrieval.final_top_k,
            "use_reranker": self.retrieval.use_reranker,
        }


# ---------------------------------------------------------------------------
# Module-level default singleton (can be overridden in tests)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = AppConfig()