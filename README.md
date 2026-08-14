# PrivaRepo 🔒

**A Fully Local AI Code Intelligence Assistant**

PrivaRepo is a production-quality Retrieval-Augmented Generation (RAG) system for code intelligence. It runs **100% locally** — no data ever leaves your machine. Index any codebase, then ask questions, search functions, explain code, find bugs, and more — all powered by local LLMs via Ollama.

---

## Architecture

```
Code Repository
      │
      ▼
Tree-sitter Parsing  ──── Python, Java, JavaScript, TypeScript
      │                   AST-boundary chunks, full metadata
      ▼
Semantic Code Chunking ── function / method / class / imports / module
      │                   Parallel, deduplicated, configurable
      ▼
Metadata Extraction ───── name, class, parent, docstring, decorators,
      │                   parameters, return type, is_async
      ▼
Embedding Generation ──── SentenceTransformer (nomic-embed-code)
      │                   Batch encoding, normalised vectors
      ▼
ChromaDB ──────────────── Persistent, local vector store
      │                   HNSW cosine similarity
      ▼
BM25 Index ────────────── rank-bm25, camelCase tokenisation
      │                   Serialised to disk, loaded on restart
      ▼
Hybrid Retrieval ──────── Vector + BM25 concurrent execution
      │
      ▼
Reciprocal Rank Fusion ── Parameter-free score merging
      │                   RRF(d) = Σ 1/(k + rank)
      ▼
Cross Encoder Reranker ── ms-marco-MiniLM-L-6-v2
      │                   Joint (query, document) pair scoring
      ▼
Prompt Builder ─────────── Grounded context injection
      │                   Task-specific templates
      ▼
Ollama LLM ─────────────── Qwen2.5-Coder:7b (local)
      │                   Streaming support, retry/backoff
      ▼
Grounded Answer ─────────── answer + reasoning + cited files + functions
```

---

## Features

| Feature | Details |
|---|---|
| **Semantic Search** | Embedding-based similarity across all code |
| **Function Search** | Filter by chunk_type=function, language, file |
| **Class Search** | Class hierarchies with parent class metadata |
| **Language Filters** | Python, Java, JavaScript, TypeScript |
| **File Filters** | Scope queries to specific files |
| **Code Explanation** | `--task explain` prompt template |
| **Bug Finding** | `--task find_bugs` prompt template |
| **Similar Code** | `--task similar_code` prompt template |
| **Interactive Chat** | Multi-turn REPL with conversation history |
| **Evaluation** | Precision@K, Recall@K, MRR, Hit Rate, Faithfulness, Relevancy |
| **Benchmarking** | P50/P95/P99 latency, memory usage, collection size |
| **Export / Import** | NDJSON format for collection portability |
| **REST API** | Optional Flask server (`privarepo serve`) |

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Parsing | tree-sitter, tree-sitter-languages | 0.21.3, 1.10.2 |
| Embeddings | sentence-transformers | 3.3.1 |
| Vector Store | chromadb | 0.5.15 |
| Sparse Retrieval | rank-bm25 | 0.2.2 |
| Reranking | sentence-transformers CrossEncoder | 3.3.1 |
| LLM | ollama | 0.4.5 |
| CLI | typer, rich | 0.12.5, 13.9.4 |
| Evaluation | RAGAS-style (local LLM judge) | — |
| Testing | pytest | 8.3.4 |
| Linting | ruff, mypy | — |

---

## Quickstart

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Install Ollama
# https://ollama.ai

# Pull the code model
ollama pull qwen2.5-coder:7b
```

### 2. Installation

```bash
git clone <repo>
cd privarepo
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Index a Repository

```bash
# Index your codebase
python cli.py index /path/to/your/project

# Index with language filter
python cli.py index /path/to/project --language python

# Reset and re-index
python cli.py index /path/to/project --reset
```

### 4. Query

```bash
# Ask a general question
python cli.py query "How does the authentication system work?"

# Explain a concept
python cli.py query "Explain the retry logic" --task explain

# Find bugs
python cli.py query "Find potential issues in the connection pool" --task find_bugs

# Filter by language and file
python cli.py query "database connections" --language java --file DatabaseService.java
```

### 5. Raw Search (No LLM)

```bash
# Hybrid search
python cli.py search "async function error handling"

# With code snippets displayed
python cli.py search "class authentication" --code

# With filters
python cli.py search "validate" --language python --type function
```

### 6. Interactive Mode

```bash
python cli.py interactive

# Inside the session:
# /search <query>    — raw search
# /stats             — collection stats
# /reset-history     — clear conversation history
# exit               — quit
```

### 7. Statistics

```bash
python cli.py stats
```

### 8. Evaluation

```bash
# Run full evaluation suite
python cli.py evaluate

# Custom queries file
python cli.py evaluate --queries data/eval_queries.json

# Save to specific path
python cli.py evaluate --output eval_results/report.json
```

### 9. Benchmark

```bash
# 20 iterations (default)
python cli.py benchmark

# Custom run count
python cli.py benchmark --runs 50 --output results/bench.json
```

### 10. Export / Import

```bash
# Export collection
python cli.py export collection_backup.ndjson

# Import (with reset)
python cli.py import collection_backup.ndjson --reset
```

### 11. REST API

```bash
python cli.py serve --host 0.0.0.0 --port 8080

# Endpoints:
# GET  /health
# GET  /stats
# POST /query  {"question": "...", "task_type": "general", "language": "python"}
# POST /search {"query": "...", "language": "python", "chunk_type": "function"}
```

---

## Configuration

All configuration is done via environment variables or a `.env` file:

```env
# Embedding
EMBEDDING_MODEL=nomic-ai/nomic-embed-code
EMBEDDING_BATCH_SIZE=64

# ChromaDB
CHROMA_PERSIST_DIR=.chromadb
CHROMA_COLLECTION=privarepo_code

# Retrieval
TOP_K_VECTOR=30
TOP_K_BM25=30
RRF_K=60
RERANK_CANDIDATES=20
FINAL_TOP_K=5
USE_RERANKER=true
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu

# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Evaluation
EVAL_K=5
BENCHMARK_ITERS=20
FAITHFULNESS_METHOD=llm
```

---

## Project Structure

```
privarepo/
├── config.py                  # Centralised typed configuration
├── tree_sitter_chunker.py     # AST-aware code parsing & chunking
├── vector_store.py            # ChromaDB + SentenceTransformer embedding
├── llm_interface.py           # Ollama client with retry/streaming
├── rag_pipeline.py            # BM25 + Vector + RRF + CrossEncoder + LLM
├── evaluator.py               # RAGAS-style evaluation suite
├── cli.py                     # Typer + Rich CLI (10 commands)
├── example_usage.py           # End-to-end demo script
├── requirements.txt           # Pinned production dependencies
├── pyproject.toml             # pytest, ruff, mypy configuration
├── data/
│   └── eval_queries.json      # 15 evaluation queries with ground truth
├── tests/
│   ├── test_chunker.py        # Tree-sitter unit tests (all 4 languages)
│   ├── test_retrieval.py      # BM25 + RRF + HybridRetriever tests
│   └── test_pipeline.py       # End-to-end integration tests
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific module
pytest tests/test_chunker.py -v
pytest tests/test_retrieval.py -v
pytest tests/test_pipeline.py -v
```

---

## Target Performance

| Metric | Target | Notes |
|---|---|---|
| Precision@5 | ≥ 70% | With 10K+ function corpus |
| Recall@5 | ≥ 95% | Hybrid retrieval advantage |
| MRR | ≥ 0.80 | Cross encoder reranking |
| Faithfulness | ≥ 0.90 | LLM-as-judge |
| Answer Relevancy | ≥ 0.90 | LLM-as-judge |
| Retrieval Latency | < 800ms | P95 |
| Generation Latency | ~2s | Qwen2.5-Coder:7b |
| Total Latency | ~2.5s | End-to-end |

---

## Design Decisions

### Why AST-Boundary Chunking (not sliding window)?

Sliding-window chunking breaks function bodies at arbitrary line counts, producing chunks that start mid-loop or mid-conditional. Tree-sitter extracts semantically complete units — the LLM always gets a complete, runnable code unit with its full context (decorators, docstring, class signature).

### Why RRF (not linear score combination)?

Linear combination requires normalising scores from two completely different scoring distributions (cosine similarity vs. BM25 TF-IDF). RRF operates on ranks, which are scale-invariant. It's parameter-free (proven effective at k=60) and handles the case where one retriever completely misses the relevant document.

### Why Cross Encoder (not just bi-encoder)?

Bi-encoder similarity scores are computed independently for query and document. Cross encoders process the (query, document) pair jointly — they can model fine-grained lexical and semantic interactions. The trade-off is O(n) inference at query time, but with only 20 candidates (post-RRF) this is ~50ms on CPU.

### Why Ollama (not API-based LLMs)?

PrivaRepo is designed for **privacy-sensitive codebases**. Source code may contain proprietary algorithms, credentials, or trade secrets. Ollama provides a locally-served LLM with no data leaving the machine. Qwen2.5-Coder is specifically trained on code and outperforms general models on code Q&A.

### Why `nomic-ai/nomic-embed-code` for embeddings?

nomic-embed-code is code-specialized and consistently outperforms all-MiniLM-L6-v2 on code retrieval benchmarks (8192 token context vs. 512). If you have tight memory constraints, fall back to `sentence-transformers/all-MiniLM-L6-v2` via `EMBEDDING_MODEL` env var.

---

## License

MIT License. See LICENSE for details.
