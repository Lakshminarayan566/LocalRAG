"""
dependencies.py — FastAPI dependency providers.

Thin wiring only: hands routers the shared PipelineManager/RepositoryManager
singletons. No business logic lives here.
"""

from __future__ import annotations

from .core.pipeline_manager import pipeline_manager, PipelineManager
from repository_manager import RepositoryManager


def get_pipeline_manager() -> PipelineManager:
    return pipeline_manager


def get_repository_manager() -> RepositoryManager:
    # Reuses the same RepositoryManager instance the PipelineManager holds,
    # so repo-list reads and repo-select writes always see one consistent
    # in-memory view of repositories.json within this process.
    return pipeline_manager.repo_manager
