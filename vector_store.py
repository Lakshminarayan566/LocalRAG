"""
vector_store.py — ChromaDB-backed Vector Store

Manages persistent storage of code chunks with embeddings and metadata.
Supports batch upsert, filtered vector search, collection statistics,
and export/import for portability.

Design decisions:
  - SentenceTransformer is loaded once and reused to avoid GPU/CPU reload overhead.
  - Batch upsert with configurable batch_size to balance memory and throughput.
  - ChromaDB metadata is a flat dict (no nested objects) — complex fields are
    serialised to strings (e.g., comma-joined decorator list).
  - Export/import uses newline-delimited JSON for streaming large collections.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import VectorStoreConfig
from tree_sitter_chunker import CodeChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Lazy singleton wrapper around SentenceTransformer.

    Lazily loads the model on first call to embed() to avoid startup
    latency when the model is not needed (e.g., import-only operations).
    """

    _instance: Optional["EmbeddingModel"] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    @classmethod
    def get(cls, model_name: str) -> "EmbeddingModel":
        if cls._instance is None or cls._model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str):
        self._model_name_str = model_name
        self._model = None  # deferred

    def _load(self) -> None:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._model_name_str)
            t0 = time.monotonic()
            self._model = SentenceTransformer(self._model_name_str)
            logger.info(
                "Embedding model loaded in %.2fs", time.monotonic() - t0
            )

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a list of texts, returns list of float vectors."""
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        self._load()
        vec = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB-backed vector store for code chunks.

    Public API:
        add_chunks(chunks)           — batch upsert with embeddings
        vector_search(query, k)      — semantic nearest-neighbour search
        get_all_documents()          — retrieve all stored documents (for BM25)
        get_stats()                  — collection statistics
        export_collection(path)      — dump to NDJSON
        import_collection(path)      — load from NDJSON
        reset()                      — drop and recreate collection
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        collection_name: Optional[str] = None,
    ):
        self.config = config or VectorStoreConfig()
        # Allows dynamically setting collection name (e.g., selected_repository.collection)
        self.collection_name = collection_name or self.config.collection_name
        self.embedding_model = EmbeddingModel.get(self.config.embedding_model)

        persist_dir = Path(self.config.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._get_or_create_collection()
        logger.info(
            "VectorStore initialised: collection='%s', dir='%s'",
            self.collection_name,
            self.config.persist_dir,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing or create new ChromaDB collection with cosine distance."""
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    def reset(self) -> None:
        """Drop and recreate the collection, clearing all indexed data."""
        logger.warning(
            "Resetting collection '%s' — all data will be lost.",
            self.collection_name,
        )
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()
        logger.info("Collection reset complete.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> int:
        """
        Embed and upsert chunks into ChromaDB in batches.

        Returns the number of chunks successfully upserted.
        Existing chunks with the same chunk_id are overwritten (upsert semantics).
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing to do.")
            return 0

        bs = batch_size or self.config.upsert_batch_size
        upserted = 0

        iterator = range(0, len(chunks), bs)
        if show_progress:
            iterator = tqdm(iterator, desc="Indexing chunks", unit="batch")

        for start in iterator:
            batch = chunks[start : start + bs]
            documents = [c.document for c in batch]
            metadatas = [c.to_metadata() for c in batch]
            ids = [c.chunk_id for c in batch]

            # Generate embeddings for this batch
            embeddings = self.embedding_model.embed(
                documents, batch_size=self.config.embedding_batch_size
            )

            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            upserted += len(batch)

        logger.info("Upserted %d chunks into collection '%s'", upserted, self.collection_name)
        return upserted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query: str,
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic nearest-neighbour search.

        Args:
            query: Natural language or code query string.
            k: Number of results to return.
            where: ChromaDB metadata filter dict (e.g., {"language": "python"}).

        Returns:
            List of dicts with keys: chunk_id, document, metadata, distance, score.
        """
        query_embedding = self.embedding_model.embed_query(query)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as exc:
            logger.error("Vector search failed: %s", exc, exc_info=True)
            return []

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance [0,2] → similarity [0,1]
                score = max(0.0, 1.0 - distance / 2.0)
                hits.append(
                    {
                        "chunk_id": chunk_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance,
                        "score": score,
                        "rank": i + 1,
                    }
                )
        return hits

    def get_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch specific chunks by their IDs."""
        if not chunk_ids:
            return []
        result = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        out = []
        for i, cid in enumerate(result["ids"]):
            out.append(
                {
                    "chunk_id": cid,
                    "document": result["documents"][i],
                    "metadata": result["metadatas"][i],
                }
            )
        return out

    def get_all_documents(
        self, batch_size: int = 1000
    ) -> List[Tuple[str, str, Dict]]:
        """
        Retrieve all (chunk_id, document, metadata) tuples from the collection.

        Used to rebuild the BM25 index after indexing.
        Streams in batches to avoid loading everything into RAM at once.
        """
        total = self._collection.count()
        if total == 0:
            return []

        all_items: List[Tuple[str, str, Dict]] = []
        offset = 0

        while offset < total:
            result = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                all_items.append((cid, doc, meta))
            offset += len(result["ids"])
            if not result["ids"]:
                break

        return all_items

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return comprehensive collection statistics."""
        count = self._collection.count()
        if count == 0:
            return {
                "total_chunks": 0,
                "collection_name": self.collection_name,
                "persist_dir": self.config.persist_dir,
                "embedding_model": self.config.embedding_model,
            }

        # Sample all metadata to compute breakdowns
        all_meta = self.get_all_documents()
        languages: Dict[str, int] = {}
        chunk_types: Dict[str, int] = {}
        files: set = set()

        for _, _, meta in all_meta:
            lang = meta.get("language", "unknown")
            ct = meta.get("chunk_type", "unknown")
            fp = meta.get("file_path", "")
            languages[lang] = languages.get(lang, 0) + 1
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
            if fp:
                files.add(fp)

        return {
            "total_chunks": count,
            "unique_files": len(files),
            "languages": languages,
            "chunk_types": chunk_types,
            "collection_name": self.collection_name,
            "persist_dir": self.config.persist_dir,
            "embedding_model": self.config.embedding_model,
            "embedding_dimension": self.embedding_model.dimension,
        }

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_collection(self, output_path: str | Path) -> int:
        """
        Export the entire collection to a newline-delimited JSON file.

        Format: one JSON object per line with keys chunk_id, document, metadata.
        Returns the number of exported records.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        items = self.get_all_documents()
        with output_path.open("w", encoding="utf-8") as fh:
            for chunk_id, document, metadata in items:
                record = {
                    "chunk_id": chunk_id,
                    "document": document,
                    "metadata": metadata,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Exported %d records to %s", len(items), output_path)
        return len(items)

    def import_collection(
        self,
        input_path: str | Path,
        batch_size: Optional[int] = None,
        reset_first: bool = False,
    ) -> int:
        """
        Import records from a previously exported NDJSON file.

        Args:
            input_path: Path to the NDJSON export file.
            batch_size: Upsert batch size.
            reset_first: If True, reset the collection before importing.

        Returns:
            Number of records imported.
        """
        if reset_first:
            self.reset()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Import file not found: {input_path}")

        bs = batch_size or self.config.upsert_batch_size
        imported = 0

        ids_batch: List[str] = []
        docs_batch: List[str] = []
        metas_batch: List[Dict] = []

        def flush_batch() -> None:
            nonlocal imported
            if not ids_batch:
                return
            embeddings = self.embedding_model.embed(
                docs_batch, batch_size=self.config.embedding_batch_size
            )
            self._collection.upsert(
                ids=ids_batch,
                documents=docs_batch,
                embeddings=embeddings,
                metadatas=metas_batch,
            )
            imported += len(ids_batch)
            ids_batch.clear()
            docs_batch.clear()
            metas_batch.clear()

        with input_path.open("r", encoding="utf-8") as fh:
            for line in tqdm(fh, desc="Importing", unit="records"):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids_batch.append(record["chunk_id"])
                docs_batch.append(record["document"])
                metas_batch.append(record["metadata"])
                if len(ids_batch) >= bs:
                    flush_batch()

        flush_batch()
        logger.info("Imported %d records from %s", imported, input_path)
        return imported

    # ------------------------------------------------------------------
    # Filtered metadata search
    # ------------------------------------------------------------------

    def metadata_filter(
        self,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a ChromaDB `where` filter dict from optional fields."""
        conditions = []
        if language:
            conditions.append({"language": {"$eq": language}})
        if chunk_type:
            conditions.append({"chunk_type": {"$eq": chunk_type}})
        if file_path:
            conditions.append({"file_path": {"$eq": file_path}})
        if class_name:
            conditions.append({"class_name": {"$eq": class_name}})
        if function_name:
            conditions.append({"function_name": {"$eq": function_name}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._collection.count()