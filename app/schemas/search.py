from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    language: Optional[str] = None
    chunk_type: Optional[str] = None
    repo: Optional[str] = Field(None, description="Repository to search. Defaults to the active repository.")


class RetrievedChunkOut(BaseModel):
    """Mirrors rag_pipeline.RetrievedChunk's public fields/properties."""

    chunk_id: str
    file_path: str
    function_name: str
    class_name: str
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    raw_code: str
    vector_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float
    final_rank: int


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunkOut]
    timings_ms: dict  # vector_ms/bm25_ms/rrf_ms/rerank_ms/total_ms, as returned by HybridRetriever.retrieve()
