# PrivaRepo API

A FastAPI layer around the existing PrivaRepo backend
(`rag_pipeline.py`, `vector_store.py`, `llm_interface.py`,
`repository_manager.py`, `tree_sitter_chunker.py`, `config.py`). This
directory contains **no retrieval, indexing, embedding, or generation
logic** — every endpoint calls into the existing classes/methods
listed below, unmodified.

## Install & run

Place `backend/` inside the existing PrivaRepo project root (alongside
`config.py`, `rag_pipeline.py`, etc.), then:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

## Endpoint → backend function map

| Endpoint | Method | Existing backend call(s) |
|---|---|---|
| `/api/health` | GET | `OllamaClient.is_available()`, `VectorStore.count`, `RepositoryManager.get_active()` |
| `/api/repositories` | POST | `RepositoryManager.add()` |
| `/api/repositories` | GET | `RepositoryManager.list()`, `RepositoryManager.get_active()` |
| `/api/repositories/active` | GET | `RepositoryManager.get_active()` |
| `/api/repositories/{name}/select` | POST | `RepositoryManager.select()` |
| `/api/repositories/{name}` | DELETE | `RepositoryManager.select()` + `RAGPipeline.reset()` (unless `?keep_data=true`), `RepositoryManager.remove()` |
| `/api/index` | POST | `RAGPipeline.index_repository()` (run as a background job) |
| `/api/index/{job_id}` | GET | (job status — in-memory, no backend call) |
| `/api/stats` | GET | `RAGPipeline.get_stats()` |
| `/api/search` | POST | `RAGPipeline.search()` |
| `/api/chat` | POST (SSE) | `RAGPipeline.search()` → `PromptBuilder.build_prompt()` → `OllamaClient.stream_generate()` (composed; see `routers/chat.py` docstring) |
| `/api/chat/sync` | POST | `RAGPipeline.query()` |
| `/api/settings` | GET | `AppConfig.summary()` |
| `/api/settings` | PATCH | `AppConfig.validate()` (via `PipelineManager.apply_settings()`) |

## Architectural constraint this API works within (not around)

`RAGPipeline.__init__()` resolves `RepositoryManager().get_active()`
internally, once, at construction — it takes no repo argument. This API
does not modify that. Instead, `core/pipeline_manager.py` caches a
single `RAGPipeline` instance and rebuilds it only when the active
repository or a mutable setting changes (calling
`RepositoryManager.select()` first, then constructing a fresh
`RAGPipeline`), exactly mirroring what the CLI does across separate
process invocations.

**Consequence:** this is a single-active-repository server, same as the
CLI. Concurrent requests for two *different* repositories will
serialize on the repo-switch, not run truly in parallel. True
concurrent multi-repo serving would require `RAGPipeline` itself to
accept an explicit repository override at construction — a backend
change, intentionally not made here.

## Not yet implemented

- **Frontend** — explicitly out of scope for this pass.
- **Auth** — none. Add before any non-local deployment.
- **Multi-worker job store** — `/api/index` job status is an in-memory
  dict; fine for a single `uvicorn` worker, not for multiple.
