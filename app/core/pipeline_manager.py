"""
core/pipeline_manager.py — Owns the RAGPipeline lifecycle for the API.

This file contains NO retrieval, indexing, or generation logic of its
own. It exists only to solve one problem created by the existing
backend's design, which this file does not modify:

    RAGPipeline.__init__() takes no repository argument — it resolves
    RepositoryManager().get_active() internally, exactly once, at
    construction time (see rag_pipeline.py's own docstring on this).
    There is no way to get a pipeline scoped to a specific, non-active
    repository without first calling RepositoryManager.select() (an
    existing public method) and then constructing a fresh RAGPipeline.

So: this manager caches ONE RAGPipeline instance (mirroring the
single-active-repository design the CLI already has — a CLI process
only ever has one active repo per invocation too), and rebuilds it
only when the active repository actually changes, or when a setting
that affects pipeline construction was mutated via /api/settings.

IMPORTANT LIMITATION (a consequence of the existing design, not
something this file works around): switching the active repository via
`get_pipeline(repo_name=...)` changes which repo is active *globally*,
same as `privarepo repo select` on the CLI. Two concurrent requests for
two DIFFERENT repos will serialize on the lock below rather than being
served truly in parallel. If concurrent multi-repo serving is needed,
RAGPipeline itself would need an explicit repo-override constructor
argument — that's a backend change and out of scope here.

Rebuilding a RAGPipeline is expensive (loads the embedding model and,
if enabled, the cross-encoder reranker — several seconds, per the
existing __init__ / _ensure_bm25_loaded code path) so this cache exists
specifically to avoid paying that cost on every request.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config import AppConfig
from rag_pipeline import RAGPipeline
from repository_manager import (
    DuplicateRepositoryError,
    InvalidRepositoryPathError,
    RepositoryManager,
    RepositoryNotFoundError,
)

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Lazily constructs and caches a RAGPipeline, rebuilding it only when
    the active repository or a pipeline-affecting setting changes.
    """

    def __init__(self) -> None:
        # One long-lived, mutable AppConfig instance. /api/settings PATCH
        # mutates fields directly on this dataclass instance (AppConfig is
        # a plain, non-frozen @dataclass — this is normal Python, not a
        # backend modification) and passes it into RAGPipeline(config=...)
        # on the next rebuild — RAGPipeline.__init__ already accepts an
        # explicit config override, this is an existing, documented
        # extension point, not new backend behaviour.
        self.cfg = AppConfig()
        self.repo_manager = RepositoryManager()

        self._pipeline: Optional[RAGPipeline] = None
        self._cached_repo_name: Optional[str] = None  # None also means "legacy no-repo mode"
        self._settings_dirty = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Pipeline access
    # ------------------------------------------------------------------

    async def get_pipeline(self, repo_name: Optional[str] = None) -> RAGPipeline:
        """
        Return a RAGPipeline scoped to `repo_name` (or the current active
        repository if `repo_name` is None), rebuilding only if needed.

        Calls: RepositoryManager.select() [if repo_name given],
               RepositoryManager.get_active(),
               RAGPipeline(config=self.cfg) [only on cache miss]
        """
        async with self._lock:
            if repo_name:
                try:
                    self.repo_manager.select(repo_name)  # existing public method
                except RepositoryNotFoundError:
                    raise  # let the router translate this to a 404

            active = self.repo_manager.get_active()  # existing public method
            active_name = active["name"] if active else None

            needs_rebuild = (
                self._pipeline is None
                or self._settings_dirty
                or active_name != self._cached_repo_name
            )
            if needs_rebuild:
                logger.info(
                    "Building RAGPipeline (repo=%s, settings_dirty=%s)",
                    active_name, self._settings_dirty,
                )
                self._pipeline = RAGPipeline(config=self.cfg)  # existing constructor, unmodified
                self._cached_repo_name = active_name
                self._settings_dirty = False

            return self._pipeline

    def invalidate(self) -> None:
        """Force the next get_pipeline() call to rebuild. Called after
        repo add/remove (collection/bm25_dir may have changed) or after
        a settings PATCH that affects pipeline construction."""
        self._settings_dirty = True

    # ------------------------------------------------------------------
    # Settings mutation
    # ------------------------------------------------------------------

    def apply_settings(self, updates: dict) -> AppConfig:
        """
        Apply a whitelisted set of dotted-path overrides (e.g.
        {"llm.max_tokens": 512, "retrieval.final_top_k": 3}) directly onto
        the live AppConfig instance, validate it, and mark the pipeline
        dirty so the change takes effect on the next request.

        Calls: AppConfig.validate() (existing method — reused as-is for
        the same misconfiguration checks the CLI relies on, e.g.
        final_top_k <= rerank_candidates).

        Raises:
            AttributeError: unknown section or field in a dotted path.
            ValueError: AppConfig.validate() rejected the resulting config
                (the mutation is rolled back before raising).
        """
        # Snapshot current values so we can roll back atomically if the
        # new combination fails validate() (e.g. final_top_k > rerank_candidates
        # after only one of the two fields was updated).
        snapshot = {}
        for dotted_path, value in updates.items():
            section_name, field_name = dotted_path.split(".", 1)
            section = getattr(self.cfg, section_name)  # raises AttributeError if unknown
            if not hasattr(section, field_name):
                raise AttributeError(f"Unknown setting: {dotted_path}")
            snapshot[dotted_path] = getattr(section, field_name)
            setattr(section, field_name, value)

        try:
            self.cfg.validate()  # existing method
        except ValueError:
            for dotted_path, old_value in snapshot.items():
                section_name, field_name = dotted_path.split(".", 1)
                setattr(getattr(self.cfg, section_name), field_name, old_value)
            raise

        self.invalidate()
        return self.cfg


# Module-level singleton — one PipelineManager per API process, shared
# across all requests via the FastAPI dependency in dependencies.py.
pipeline_manager = PipelineManager()
