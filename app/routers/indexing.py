"""
routers/indexing.py

Calls:
  - RepositoryManager.get_repository() / get_active()  -> resolve which path to index
  - RAGPipeline.index_repository()                       -> POST /api/index (the actual work)
  - RAGPipeline.get_stats()                               -> GET  /api/stats

Indexing a real repository can take a while (chunking + embedding +
BM25 build — see rag_pipeline.py's own index_repository() timing
breakdown). Running that inline on a request thread would block the
event loop and time out most HTTP clients, so this router runs it via
FastAPI's BackgroundTasks and exposes a job-status endpoint to poll.
This orchestration is new code; the indexing work itself is a single,
unmodified call to RAGPipeline.index_repository().
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
from ..schemas.indexing import IndexJobStatus, IndexRequest, StatsResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["indexing"])

# In-memory job store. A single-process API is assumed (matches the
# existing backend's single-process, single-active-repository design —
# see pipeline_manager.py's module docstring). A multi-worker deployment
# would need this moved to shared storage (e.g. Redis), which is a
# deployment concern, not a change to the indexing logic itself.
_jobs: Dict[str, IndexJobStatus] = {}


def _run_index_job(job_id: str, repo_path: str, languages: Optional[list], mgr: PipelineManager) -> None:
    """Runs on FastAPI's background task thread pool."""
    try:
        # get_pipeline() is async (it takes an asyncio.Lock); this callback
        # runs in a plain thread via BackgroundTasks, so drive it through
        # asyncio.run() rather than awaiting directly.
        import asyncio
        pipeline = asyncio.run(mgr.get_pipeline())
        include_ext = None
        if languages:
            include_ext = [
                ext for ext, lang in pipeline.config.chunker.supported_extensions.items()
                if lang in languages
            ]
        result = pipeline.index_repository(  # RAGPipeline.index_repository()
            repo_path=repo_path,
            include_extensions=include_ext,
            show_progress=False,
        )
        _jobs[job_id] = IndexJobStatus(
            job_id=job_id, status="completed", repo=repo_path, result=result
        )
    except Exception as exc:
        logger.exception("Indexing job %s failed", job_id)
        _jobs[job_id] = IndexJobStatus(
            job_id=job_id, status="failed", repo=repo_path, error=str(exc)
        )


@router.post("/index", response_model=IndexJobStatus, status_code=202)
async def start_indexing(
    body: IndexRequest,
    background_tasks: BackgroundTasks,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> IndexJobStatus:
    """Kicks off indexing as a background job and returns immediately with a job_id to poll."""
    if body.repo:
        entry = mgr.repo_manager.get_repository(body.repo)  # RepositoryManager.get_repository()
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No registered repository named '{body.repo}'.")
        repo_path = entry["path"]
        # Ensure this repo is the active one, since index_repository() is
        # called on whatever pipeline get_pipeline() returns, which is
        # always scoped to the active repository (see pipeline_manager.py).
        if mgr.repo_manager.get_active() is None or mgr.repo_manager.get_active()["name"] != entry["name"]:
            mgr.repo_manager.select(entry["name"])  # RepositoryManager.select()
            mgr.invalidate()
    else:
        active = mgr.repo_manager.get_active()  # RepositoryManager.get_active()
        if active is None:
            raise HTTPException(
                status_code=400,
                detail="No repo specified and no active repository set. "
                       "POST /api/repositories first, or pass 'repo' explicitly.",
            )
        repo_path = active["path"]

    job_id = str(uuid.uuid4())
    _jobs[job_id] = IndexJobStatus(job_id=job_id, status="running", repo=repo_path)
    background_tasks.add_task(_run_index_job, job_id, repo_path, body.languages, mgr)
    return _jobs[job_id]


@router.get("/index/{job_id}", response_model=IndexJobStatus)
async def get_index_status(job_id: str) -> IndexJobStatus:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No indexing job with id '{job_id}'.")
    return job


@router.get("/stats", response_model=StatsResponse)
async def get_stats(mgr: PipelineManager = Depends(get_pipeline_manager)) -> StatsResponse:
    """Calls: RAGPipeline.get_stats()."""
    pipeline = await mgr.get_pipeline()
    stats = pipeline.get_stats()
    return StatsResponse(**stats)
