"""
routers/repositories.py

Calls:
  - RepositoryManager.add()          -> POST /api/repositories
  - RepositoryManager.list()         -> GET  /api/repositories
  - RepositoryManager.get_active()   -> GET  /api/repositories/active
  - RepositoryManager.select()       -> POST /api/repositories/{name}/select
  - RAGPipeline.reset()              -> DELETE /api/repositories/{name} (unless keep_data=true)
  - RepositoryManager.remove()       -> DELETE /api/repositories/{name}

No retrieval/indexing/embedding logic lives here — this router only
orchestrates existing RepositoryManager/RAGPipeline methods.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
from rag_pipeline import RAGPipeline
from repository_manager import (
    DuplicateRepositoryError,
    InvalidRepositoryPathError,
    RepositoryNotFoundError,
)
from ..schemas.repository import RepositoryAddRequest, RepositoryListResponse, RepositoryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/repositories", tags=["repositories"])


def _to_response(entry: dict, active_name: str | None) -> RepositoryResponse:
    return RepositoryResponse(
        name=entry["name"],
        path=entry["path"],
        collection=entry["collection"],
        bm25_dir=entry["bm25_dir"],
        is_active=(entry["name"] == active_name),
    )


@router.post("", response_model=RepositoryResponse, status_code=201)
async def add_repository(
    body: RepositoryAddRequest,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> RepositoryResponse:
    """Register a repository. Calls: RepositoryManager.add(path, name)."""
    try:
        entry = mgr.repo_manager.add(body.path, name=body.name)
    except InvalidRepositoryPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DuplicateRepositoryError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    mgr.invalidate()  # a new repo may have just become active (first repo registered)
    active = mgr.repo_manager.get_active()
    return _to_response(entry, active["name"] if active else None)


@router.get("", response_model=RepositoryListResponse)
async def list_repositories(
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> RepositoryListResponse:
    """Calls: RepositoryManager.list(), RepositoryManager.get_active()."""
    active = mgr.repo_manager.get_active()
    active_name = active["name"] if active else None
    return RepositoryListResponse(
        active=active_name,
        repositories=[_to_response(r, active_name) for r in mgr.repo_manager.list()],
    )


@router.get("/active", response_model=Optional[RepositoryResponse])
async def get_active_repository(
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> Optional[RepositoryResponse]:
    """Calls: RepositoryManager.get_active()."""
    active = mgr.repo_manager.get_active()
    if active is None:
        return None
    return _to_response(active, active["name"])


@router.post("/{name}/select", response_model=RepositoryResponse)
async def select_repository(
    name: str,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> RepositoryResponse:
    """Calls: RepositoryManager.select(name). Rebuilds the cached pipeline
    on next use — this endpoint doesn't rebuild eagerly, so it returns
    immediately even though the switch's real cost lands on the next
    /chat or /search call."""
    try:
        entry = mgr.repo_manager.select(name)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    mgr.invalidate()
    return _to_response(entry, entry["name"])


@router.delete("/{name}", status_code=204)
async def remove_repository(
    name: str,
    keep_data: bool = False,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> None:
    """
    Unregister a repository. By default also deletes its indexed data.

    Calls: RepositoryManager.get_repository() [existence check],
           RepositoryManager.select() + RAGPipeline(config=...).reset()
               [unless keep_data=true],
           RepositoryManager.remove()
    """
    entry = mgr.repo_manager.get_repository(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No registered repository named '{name}'.")

    if not keep_data:
        # Must select() it first — RAGPipeline only ever resolves the
        # *active* repository at construction time (see pipeline_manager.py's
        # module docstring), so there's no other way to build a pipeline
        # scoped to a specific, possibly-non-active repo for cleanup.
        mgr.repo_manager.select(name)
        try:
            RAGPipeline(config=mgr.cfg).reset()  # existing constructor + existing reset() method
        except Exception as exc:
            logger.warning("Could not clean up indexed data for '%s': %s", name, exc)

    try:
        mgr.repo_manager.remove(name)
    except RepositoryNotFoundError as exc:
        # Can't realistically happen given the existence check above, but
        # handled for correctness under concurrent requests.
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    mgr.invalidate()
