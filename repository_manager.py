"""
repository_manager.py — Multi-repository registry for PrivaRepo

Phase 1: pure bookkeeping. Owns repositories.json and tracks registered
{name, path, collection, bm25_dir} entries plus which one is "active".

Deliberately does not create, delete, or touch any ChromaDB collection or
BM25 index on disk — `collection` and `bm25_dir` here are just generated
identifiers for vector_store.py / rag_pipeline.py to consume in a later
phase. This module has no dependency on them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).parent.resolve()
_DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / "repositories.json"
_DEFAULT_BM25_BASE_DIR = _PROJECT_ROOT / ".bm25"


# ---------------------------------------------------------------------------
# Exceptions
#
# Each also inherits from the "closest" matching builtin exception type, so
# callers can catch either the specific type or the familiar builtin one
# (e.g. `except ValueError` for a duplicate name, `except KeyError` for a
# missing repository) without needing to know about this module's types.
# ---------------------------------------------------------------------------

class RepositoryError(Exception):
    """Base class for all repository-registry errors."""


class DuplicateRepositoryError(RepositoryError, ValueError):
    """Raised by add() when the given name is already registered."""


class RepositoryNotFoundError(RepositoryError, KeyError):
    """Raised by select()/remove() when the given name isn't registered."""

    def __str__(self) -> str:
        # KeyError.__str__ reprs its first arg (adds quotes) — override so
        # str(exc) reads as a normal message instead.
        return Exception.__str__(self)


class InvalidRepositoryPathError(RepositoryError, NotADirectoryError):
    """Raised by add() when the given path doesn't exist or isn't a directory."""


# ---------------------------------------------------------------------------
# RepositoryManager
# ---------------------------------------------------------------------------

class RepositoryManager:
    """
    Manages the multi-repository registry persisted in repositories.json.

    Usage:
        mgr = RepositoryManager()
        entry = mgr.add("/path/to/repo", name="my-repo")
        mgr.select("my-repo")
        mgr.list()
        mgr.get_active()
        mgr.remove("my-repo")
    """

    def __init__(self, registry_path: Optional[str] = None):
        self._path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY_PATH
        self._data = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if not self._path.exists():
            return {"active": None, "repositories": []}
        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Tolerate a partially-formed file rather than raising KeyError deep
        # inside some later method.
        data.setdefault("active", None)
        data.setdefault("repositories", [])
        return data

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
            f.write("\n")

    # ------------------------------------------------------------------
    # Naming helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slugify(name: str) -> str:
        """Turn a display name into a safe collection-name slug: lowercase,
        alnum + underscores, no leading/trailing/repeated underscores."""
        raw = "".join(c.lower() if c.isalnum() else "_" for c in name)
        slug = "_".join(part for part in raw.split("_") if part)
        return slug or "repo"

    def _unique_collection_name(self, base_slug: str) -> str:
        """Disambiguate on collision, e.g. two repos both named 'api'."""
        existing = {r["collection"] for r in self._data["repositories"]}
        collection = base_slug
        i = 2
        while collection in existing:
            collection = f"{base_slug}_{i}"
            i += 1
        return collection

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_repository(self, name: str) -> Optional[dict]:
        """Case-insensitive lookup by name. Returns None if not found."""
        name_l = name.lower()
        for r in self._data["repositories"]:
            if r["name"].lower() == name_l:
                return r
        return None

    def list(self) -> list[dict]:
        """Return all registered repository entries, in registration order."""
        return list(self._data["repositories"])

    def get_active(self) -> Optional[dict]:
        """Return the active repository's entry, or None if none is active."""
        active_name = self._data.get("active")
        if not active_name:
            return None
        return self.get_repository(active_name)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, path: str, name: Optional[str] = None) -> dict:
        """
        Register a repository.

        Raises:
            InvalidRepositoryPathError: path doesn't exist / isn't a directory.
            DuplicateRepositoryError: the resolved name is already registered.
        """
        repo_path = Path(path).resolve()
        if not repo_path.is_dir():
            raise InvalidRepositoryPathError(
                f"'{repo_path}' is not a valid directory."
            )

        display_name = name or repo_path.name

        if self.get_repository(display_name) is not None:
            raise DuplicateRepositoryError(
                f"A repository named '{display_name}' is already registered. "
                "Pass a different `name` to register it under another name."
            )

        collection = self._unique_collection_name(self._slugify(display_name))
        bm25_dir = str(_DEFAULT_BM25_BASE_DIR / collection)  # unique: derived from a unique collection name

        entry = {
            "name": display_name,
            "path": str(repo_path),
            "collection": collection,
            "bm25_dir": bm25_dir,
        }
        self._data["repositories"].append(entry)
        self._save()
        return entry

    def select(self, name: str) -> dict:
        """
        Mark a registered repository as active.

        Raises:
            RepositoryNotFoundError: no repository named `name` is registered.
        """
        entry = self.get_repository(name)
        if entry is None:
            raise RepositoryNotFoundError(f"No registered repository named '{name}'.")

        self._data["active"] = entry["name"]
        self._save()
        return entry

    def remove(self, name: str) -> None:
        """
        Unregister a repository. Pure bookkeeping — does not delete any
        indexed data. If the removed repository was active, the active
        pointer moves to the next remaining repository (registration order),
        or to None if none remain.

        Raises:
            RepositoryNotFoundError: no repository named `name` is registered.
        """
        entry = self.get_repository(name)
        if entry is None:
            raise RepositoryNotFoundError(f"No registered repository named '{name}'.")

        self._data["repositories"] = [
            r for r in self._data["repositories"]
            if r["name"].lower() != entry["name"].lower()
        ]

        if self._data.get("active") and self._data["active"].lower() == entry["name"].lower():
            remaining = self._data["repositories"]
            self._data["active"] = remaining[0]["name"] if remaining else None

        self._save()