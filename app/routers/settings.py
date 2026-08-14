"""
routers/settings.py

Calls:
  - AppConfig.summary() (config.py) — GET, as the base, extended with a
    couple of extra fields summary() doesn't include (max_tokens, keep_alive)
  - PipelineManager.apply_settings() — PATCH, which itself calls
    AppConfig.validate() (config.py, existing method, unmodified) before
    committing any change.

Note: only a whitelisted subset of AppConfig is exposed as mutable (see
schemas/settings.py's MUTABLE_FIELDS) — structural fields owned by
repository scoping (persist_dir, collection_name, bm25_index_dir) are
intentionally excluded, since those are managed by RAGPipeline.__init__
via repository_manager.py, not by runtime settings.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
from ..schemas.settings import SettingsResponse, SettingsUpdateRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings", tags=["settings"])


def _to_response(mgr: PipelineManager) -> SettingsResponse:
    summary = mgr.cfg.summary()  # AppConfig.summary() — existing method
    return SettingsResponse(
        embedding_model=summary["embedding_model"],
        collection=summary["collection"],
        persist_dir=summary["persist_dir"],
        llm_model=summary["llm_model"],
        llm_max_tokens=mgr.cfg.llm.max_tokens,
        llm_temperature=mgr.cfg.llm.temperature,
        llm_keep_alive=mgr.cfg.llm.keep_alive,
        top_k_vector=summary["top_k_vector"],
        top_k_bm25=summary["top_k_bm25"],
        rrf_k=summary["rrf_k"],
        rerank_candidates=summary["rerank_candidates"],
        final_top_k=summary["final_top_k"],
        use_reranker=summary["use_reranker"],
    )


@router.get("", response_model=SettingsResponse)
async def get_settings(mgr: PipelineManager = Depends(get_pipeline_manager)) -> SettingsResponse:
    return _to_response(mgr)


@router.patch("", response_model=SettingsResponse)
async def update_settings(
    body: SettingsUpdateRequest,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> SettingsResponse:
    """
    Applies only the fields provided. Changes take effect on the NEXT
    /chat or /search request, not this one — rebuilding a RAGPipeline
    (re-loading the embedding model / cross-encoder) is expensive, so
    this endpoint marks the cache dirty rather than rebuilding eagerly.
    """
    updates = body.to_dotted_updates()
    if not updates:
        raise HTTPException(status_code=400, detail="No settings fields provided.")

    try:
        mgr.apply_settings(updates)  # PipelineManager.apply_settings()
    except AttributeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # AppConfig.validate() rejected the resulting combination
        # (e.g. final_top_k > rerank_candidates)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _to_response(mgr)
