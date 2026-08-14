from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by every endpoint on failure."""

    error: str = Field(..., description="Short machine-oriented error code")
    message: str = Field(..., description="Human-readable explanation")
    detail: Optional[str] = Field(None, description="Extra context (e.g. underlying exception text)")


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    ollama_reachable: bool
    ollama_model: str
    active_repository: Optional[str] = None
    indexed_chunks: Optional[int] = None
