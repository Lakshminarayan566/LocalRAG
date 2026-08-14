from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .search import RetrievedChunkOut

VALID_TASK_TYPES = (
    "general", "explain", "find_bugs",
    "similar_code", "function_search", "class_search",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    task_type: str = Field("general", description=f"One of: {', '.join(VALID_TASK_TYPES)}")
    language: Optional[str] = None
    chunk_type: Optional[str] = None
    file_path: Optional[str] = None
    class_name: Optional[str] = None
    repo: Optional[str] = Field(None, description="Repository to query. Defaults to the active repository.")


class ChatResponse(BaseModel):
    """Mirrors rag_pipeline.RAGResponse exactly (non-streaming endpoint)."""

    question: str
    answer: str
    reasoning: str
    referenced_files: List[str]
    functions_used: List[str]
    retrieved_chunks: List[RetrievedChunkOut]
    retrieval_time: float
    generation_time: float
    total_time: float
    model: str
    filters_applied: dict
