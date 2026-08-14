"""
routers/search.py

Calls:
  - RAGPipeline.search()   -> the entire endpoint; no LLM call, no
    reimplementation of retrieval/RRF/reranking (that all lives in
    HybridRetriever.retrieve(), called internally by RAGPipeline.search()).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
from repository_manager import RepositoryNotFoundError
from ..schemas.search import RetrievedChunkOut, SearchRequest, SearchResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


def _chunk_to_out(chunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        function_name=chunk.function_name,
        class_name=chunk.class_name,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        raw_code=chunk.raw_code,
        vector_score=chunk.vector_score,
        bm25_score=chunk.bm25_score,
        rrf_score=chunk.rrf_score,
        rerank_score=chunk.rerank_score,
        final_rank=chunk.final_rank,
    )


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    mgr: PipelineManager = Depends(get_pipeline_manager),
) -> SearchResponse:
    try:
        pipeline = await mgr.get_pipeline(repo_name=body.repo)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    chunks, timings = pipeline.search(  # RAGPipeline.search()
        query=body.query,
        language=body.language,
        chunk_type=body.chunk_type,
    )
    return SearchResponse(
        query=body.query,
        results=[_chunk_to_out(c) for c in chunks],
        timings_ms=timings,
    )
