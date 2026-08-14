"""
routers/health.py

Calls:
  - OllamaClient.is_available()   (llm_interface.py, via pipeline.llm)
  - VectorStore.count             (vector_store.py, via pipeline.vector_store, a @property)
  - RepositoryManager.get_active()(repository_manager.py)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
from ..schemas.common import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(mgr: PipelineManager = Depends(get_pipeline_manager)) -> HealthResponse:
    """
    Reports whether Ollama is reachable with the configured model, and
    basic index status for the active repository. Never raises — a
    down Ollama or empty index is a "degraded" status, not an error,
    since the API itself is still up.
    """
    active = mgr.repo_manager.get_active()  # RepositoryManager.get_active()

    try:
        pipeline = await mgr.get_pipeline()
        ollama_ok = pipeline.llm.is_available()  # OllamaClient.is_available()
        indexed_chunks = pipeline.vector_store.count  # VectorStore.count property
    except Exception as exc:
        logger.warning("Health check could not build a pipeline: %s", exc)
        return HealthResponse(
            status="degraded",
            ollama_reachable=False,
            ollama_model=mgr.cfg.llm.model,
            active_repository=active["name"] if active else None,
            indexed_chunks=None,
        )

    return HealthResponse(
        status="ok" if ollama_ok else "degraded",
        ollama_reachable=ollama_ok,
        ollama_model=mgr.cfg.llm.model,
        active_repository=active["name"] if active else None,
        indexed_chunks=indexed_chunks,
    )
