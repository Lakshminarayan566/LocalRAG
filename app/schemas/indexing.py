from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    repo: Optional[str] = Field(
        None, description="Registered repository name to index. Defaults to the active repository."
    )
    languages: Optional[List[str]] = Field(
        None, description="Restrict indexing to these languages, e.g. ['python', 'java']"
    )


class IndexJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed"
    repo: Optional[str]
    result: Optional[dict] = None  # RAGPipeline.index_repository()'s return dict, once completed
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """
    Mirrors RAGPipeline.get_stats()'s return shape exactly, which itself
    is VectorStore.get_stats() plus bm25_index_size/bm25_index_built.
    All fields but total_chunks are Optional because VectorStore.get_stats()
    returns a smaller dict when the collection is empty (count == 0).
    """

    total_chunks: int
    collection_name: Optional[str] = None
    persist_dir: Optional[str] = None
    embedding_model: Optional[str] = None
    unique_files: Optional[int] = None
    languages: Optional[dict] = None
    chunk_types: Optional[dict] = None
    embedding_dimension: Optional[int] = None
    bm25_index_size: Optional[int] = None
    bm25_index_built: Optional[bool] = None
