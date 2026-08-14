from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RepositoryAddRequest(BaseModel):
    path: str = Field(..., description="Absolute filesystem path to the repository")
    name: Optional[str] = Field(None, description="Display name (defaults to the folder name)")


class RepositoryResponse(BaseModel):
    name: str
    path: str
    collection: str
    bm25_dir: str
    is_active: bool


class RepositoryListResponse(BaseModel):
    active: Optional[str]
    repositories: List[RepositoryResponse]
