"""
rag_pipeline.py — Hybrid Retrieval + Generation Pipeline

Implements the full RAG pipeline:
  1. Hybrid retrieval: Vector Search + BM25 (concurrent)
  2. Reciprocal Rank Fusion (RRF) for score merging
  3. Cross Encoder reranking
  4. Prompt construction with grounded context
  5. LLM generation with structured response

Design decisions:
  - RRF chosen over linear score combination because it's rank-based and
    robust to score distribution mismatches between dense and sparse models
    (no need to normalise cosine similarity against BM25 scores). Per-list
    weights (vector_weight, bm25_weight; default 0.7/0.3) let the vector
    and BM25 signals contribute unequally to fusion while keeping the
    rank-based robustness — equal weights (1.0/1.0) reproduce standard
    unweighted RRF exactly.
  - BM25 index is serialised to disk as a pickle to avoid full reindex on restart.
  - Cross encoder runs on CPU batch inference; batch_size=16 is optimal for
    ms-marco-MiniLM-L-6-v2 on a 4-core machine.
  - PromptBuilder enforces strict grounding: the system prompt explicitly
    prohibits hallucination and mandates source attribution.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import AppConfig, RetrievalConfig
from llm_interface import LLMResponse, OllamaClient
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A code chunk retrieved and scored by the hybrid pipeline."""

    chunk_id: str
    document: str
    metadata: Dict[str, Any]

    # Individual retriever scores
    vector_score: float = 0.0     # Cosine similarity [0, 1]
    bm25_score: float = 0.0       # Normalised BM25 score [0, 1]
    vector_rank: int = 0          # Rank in vector results (1-indexed)
    bm25_rank: int = 0            # Rank in BM25 results (1-indexed)

    # Fusion and reranking
    rrf_score: float = 0.0        # Reciprocal Rank Fusion score
    rerank_score: float = 0.0     # Cross encoder logit score
    final_rank: int = 0

    @property
    def file_path(self) -> str:
        return self.metadata.get("file_path", "")

    @property
    def function_name(self) -> str:
        return self.metadata.get("function_name", "")

    @property
    def class_name(self) -> str:
        return self.metadata.get("class_name", "")

    @property
    def language(self) -> str:
        return self.metadata.get("language", "")

    @property
    def chunk_type(self) -> str:
        return self.metadata.get("chunk_type", "")

    @property
    def start_line(self) -> int:
        return int(self.metadata.get("start_line", 0))

    @property
    def end_line(self) -> int:
        return int(self.metadata.get("end_line", 0))

    @property
    def raw_code(self) -> str:
        """Extract just the code portion from the enriched document."""
        lines = self.document.split("\n")
        # Skip the header lines added by _build_document
        for i, line in enumerate(lines):
            if line.strip() == "":
                return "\n".join(lines[i + 1:]).strip()
        return self.document


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline."""

    question: str
    answer: str
    reasoning: str
    referenced_files: List[str]
    functions_used: List[str]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_time: float      # seconds
    generation_time: float     # seconds
    total_time: float          # seconds
    model: str
    filters_applied: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Persistent BM25 index built from the ChromaDB collection.

    Tokens are extracted from the enriched document text.
    The index is serialised to disk and reloaded on startup.
    """

    _TOKENIZE_RE = __import__("re").compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.index_dir / "bm25_index.pkl"
        self._corpus_path = self.index_dir / "bm25_corpus.pkl"

        self._bm25: Optional[BM25Okapi] = None
        self._corpus_ids: List[str] = []          # chunk_id for each corpus doc
        self._corpus_docs: List[str] = []          # raw document text
        self._corpus_metas: List[Dict] = []        # metadata for each doc

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def size(self) -> int:
        return len(self._corpus_ids)

    def build(
        self,
        items: List[Tuple[str, str, Dict]],
        save: bool = True,
    ) -> None:
        """
        Build the BM25 index from (chunk_id, document, metadata) tuples.
        Overwrites any previously built index.
        """
        if not items:
            logger.warning("BM25 build called with empty corpus.")
            return

        logger.info("Building BM25 index over %d documents...", len(items))
        t0 = time.monotonic()

        self._corpus_ids = [item[0] for item in items]
        self._corpus_docs = [item[1] for item in items]
        self._corpus_metas = [item[2] for item in items]

        tokenised = [self._tokenize(doc) for doc in self._corpus_docs]
        self._bm25 = BM25Okapi(tokenised)

        elapsed = time.monotonic() - t0
        logger.info("BM25 index built in %.2fs (%d docs)", elapsed, len(items))

        if save:
            self._save()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenise document text for BM25.
        Extracts identifier-like tokens (camelCase, snake_case, etc.)
        and splits camelCase into sub-tokens for better recall.
        """
        raw_tokens = self._TOKENIZE_RE.findall(text)
        # Split camelCase: "myFunction" → ["my", "function"]
        expanded: List[str] = []
        for tok in raw_tokens:
            sub = __import__("re").sub(r"([a-z])([A-Z])", r"\1 \2", tok)
            expanded.extend(sub.lower().split())
        return expanded or [""]

    def search(
        self,
        query: str,
        k: int = 10,
        language_filter: Optional[str] = None,
        chunk_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the BM25 index for the top-K matching documents.

        Optionally filters by language or chunk_type post-scoring.
        Returns list of dicts with keys: chunk_id, document, metadata, score, rank.
        """
        if not self.is_built:
            logger.warning("BM25 index not built — returning empty results.")
            return []

        query_tokens = self._tokenize(query)
        raw_scores = self._bm25.get_scores(query_tokens)

        # Normalise scores to [0, 1]
        max_score = float(np.max(raw_scores)) if raw_scores.any() else 1.0
        if max_score == 0:
            max_score = 1.0

        # Build scored results with optional metadata filtering
        scored: List[Tuple[float, int]] = []
        for idx, score in enumerate(raw_scores):
            if language_filter and self._corpus_metas[idx].get("language") != language_filter:
                continue
            if chunk_type_filter and self._corpus_metas[idx].get("chunk_type") != chunk_type_filter:
                continue
            scored.append((float(score), idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        results = []
        for rank, (raw_score, idx) in enumerate(top, start=1):
            results.append(
                {
                    "chunk_id": self._corpus_ids[idx],
                    "document": self._corpus_docs[idx],
                    "metadata": self._corpus_metas[idx],
                    "score": raw_score / max_score,
                    "raw_score": raw_score,
                    "rank": rank,
                }
            )
        return results

    def _save(self) -> None:
        """Serialise index and corpus to disk."""
        with self._index_path.open("wb") as f:
            pickle.dump(self._bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        with self._corpus_path.open("wb") as f:
            pickle.dump(
                (self._corpus_ids, self._corpus_docs, self._corpus_metas), f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.debug("BM25 index saved to %s", self.index_dir)

    def load(self) -> bool:
        """
        Load a previously saved index from disk.
        Returns True if successful, False if no saved index found.
        """
        if not (self._index_path.exists() and self._corpus_path.exists()):
            return False
        try:
            with self._index_path.open("rb") as f:
                self._bm25 = pickle.load(f)
            with self._corpus_path.open("rb") as f:
                self._corpus_ids, self._corpus_docs, self._corpus_metas = pickle.load(f)
            logger.info(
                "BM25 index loaded from disk (%d documents)", len(self._corpus_ids)
            )
            return True
        except Exception as exc:
            logger.error("Failed to load BM25 index: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: List[Dict[str, Any]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using (optionally weighted)
    Reciprocal Rank Fusion.

    RRF score for document d:
        RRF(d) = Σ_r w_r · 1 / (k + rank_r(d))

    where rank_r(d) is the 1-indexed rank of d in result list r, and w_r is
    the weight assigned to that list (defaults to 1.0 for every list,
    reproducing standard unweighted RRF exactly).

    Documents not present in a list simply contribute 0 from that list —
    there is no explicit "missing" penalty term; this matches the
    original RRF formulation and the pre-existing behaviour of this
    function.

    Args:
        *ranked_lists: Variable number of ranked result lists.
                       Each list is [{chunk_id, document, metadata, score, rank}, ...].
        k: RRF constant (default 60 per the original paper).
        weights: Optional per-list weights, same order/length as
                 *ranked_lists. Defaults to equal weight (1.0) per list —
                 i.e. identical to standard RRF — when omitted.

    Returns:
        Merged list sorted by RRF score descending, with injected rrf_score field.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match the number of "
            f"ranked_lists ({len(ranked_lists)})"
        )

    rrf_scores: Dict[str, float] = {}
    chunk_data: Dict[str, Dict] = {}

    for weight, result_list in zip(weights, ranked_lists):
        for item in result_list:
            cid = item["chunk_id"]
            rank = item.get("rank", len(result_list))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight * (1.0 / (k + rank))
            if cid not in chunk_data:
                chunk_data[cid] = item

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for rank, cid in enumerate(sorted_ids, start=1):
        item = dict(chunk_data[cid])
        item["rrf_score"] = rrf_scores[cid]
        item["rank"] = rank
        merged.append(item)

    return merged


# ---------------------------------------------------------------------------
# Cross Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Cross Encoder reranker using ms-marco-MiniLM-L-6-v2.

    The cross encoder scores (query, document) pairs jointly, giving
    much more accurate relevance scores than bi-encoder cosine similarity.
    Trade-off: O(n × query) inference vs O(1) for bi-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        logger.info("Loading cross encoder: %s on %s", model_name, device)
        self._model = CrossEncoder(model_name, device=device)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
        batch_size: int = 16,
        score_threshold: float = -10.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using (query, document) pair scoring.

        Args:
            query: The user query.
            candidates: List of candidate dicts (must have 'document' key).
            top_n: Number of top results to return after reranking.
            batch_size: Inference batch size.
            score_threshold: Minimum score to include a candidate.

        Returns:
            Top-N candidates sorted by rerank_score descending.
        """
        if not candidates:
            return []

        pairs = [(query, c["document"]) for c in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        scored = []
        for candidate, score in zip(candidates, scores):
            if float(score) >= score_threshold:
                item = dict(candidate)
                item["rerank_score"] = float(score)
                scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        top = scored[:top_n]
        for rank, item in enumerate(top, start=1):
            item["rank"] = rank

        return top


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Constructs grounded prompts for the Ollama LLM.

    Task-specific templates ensure the LLM focuses on the right kind of
    analysis (explanation, bug-finding, similarity search, etc.).
    """

    SYSTEM_PROMPT = """\
You are PrivaRepo, an expert AI code assistant with deep knowledge of software engineering.

STRICT RULES — YOU MUST FOLLOW THESE WITHOUT EXCEPTION:
1. ONLY answer using the code context provided below. Do not use any outside knowledge.
2. If the provided context is insufficient to answer the question, say:
   "I cannot answer this question based on the available code context."
   If it is only PARTIALLY sufficient, answer what you can and then explicitly
   state what cannot be established from the retrieved code.
3. NEVER hallucinate function names, file paths, class names, or code behaviour.
4. Always cite the exact file path and function/class name when referencing code,
   and include line ranges whenever they are present in the retrieved context.
5. GROUNDING — the retrieved CODE CONTEXT is your sole source of truth for how
   this repository works:
   - Do not use general textbook, academic, or "how this algorithm usually
     works" knowledge to fill in implementation details the context doesn't
     show. If the context doesn't show it, it doesn't go in the answer.
   - Explicitly separate the name of an algorithm/technique (e.g. "BM25",
     "RRF") from this repository's actual implementation of it. You may name
     the algorithm, but describe its behaviour only from what the retrieved
     code does — never from how that algorithm is typically defined elsewhere.
   - If the retrieved context shows two components being combined or used
     together (e.g. two retrieval stages), describe exactly what the code
     does with them (what calls what, what data flows where) — do not add
     the underlying mathematical or theoretical explanation of either
     component unless that theory is explicitly written in the retrieved
     code (e.g. in a comment or docstring).
   - Do not describe vector search as part of BM25 (or vice versa) if the
     code implements and calls them as separate retrieval stages that are
     later combined — describe them as separate stages, exactly as the code
     shows.
   - Do not call a score-combination step a "weighted average" (or any other
     specific mathematical operation) unless the retrieved code explicitly
     implements that operation. If the code just calls a function/method
     without showing its internal formula, say the combination happens via
     that named function and that its internal scoring logic is not
     established by the retrieved context.
   - Prefer exact function calls, classes, files, and data flow over any
     restatement of general theory.
   - When uncertain whether a detail (a formula, a scoring weight, an
     internal step) is actually shown in the retrieved context, do not guess
     or infer it — say plainly: "The retrieved code does not establish this
     detail."
6. Structure your response EXACTLY as follows:

## ANSWER
<Lead with a direct, one-to-two sentence answer to the question. Then explain
it in clear language for someone studying this repository for the first time,
using short paragraphs (and headings/sub-headings if the question has multiple
parts). Ground everything in the retrieved context — reference this repository's
actual files, classes, and functions, not generic textbook explanations.

Scale the depth of the explanation to the question:
- Simple factual question ("what does X return", "where is Y defined") →
  1-3 paragraphs is enough. Don't pad it.
- Implementation / "how does X work" question → walk through the flow
  step-by-step (a short numbered list is fine), covering WHAT the relevant
  code does and HOW the pieces fit together, in roughly 3-6 paragraphs.
- Architecture / component question → start with a brief overview, then
  describe each relevant component and how they connect to one another,
  citing the specific classes/functions that implement each connection.

Where the context supports it, explain WHY a component exists (its purpose
in the system), not just what it does mechanically. Never invent behaviour,
files, or relationships that aren't shown in the CODE CONTEXT — if something
relevant is missing from the context, say so plainly instead of guessing.
Keep the tone professional and technically precise throughout.>

## REASONING
<Concise evidence trail, not private chain-of-thought: list the key facts from
the CODE CONTEXT that support the answer above (e.g. "X calls Y at line N,
which then..."). A few short bullets or sentences is enough — this is a
summary of the evidence you relied on, not a transcript of your thinking.>

## REFERENCED FILES
- <file_path_1>
- <file_path_2>

## FUNCTIONS USED
- <function_name> (<file_path>)

TASK-SPECIFIC RULES:

FIND_BUGS:
- Report ONLY bugs directly demonstrated by the retrieved code.
- A missing validation check is NOT a bug by itself.
- Do NOT invent attacks, crashes, security problems, or failure scenarios.
- For every reported bug, provide:
  1. File/function
  2. Exact code condition
  3. Concrete resulting failure
- If you cannot prove a bug from the retrieved code, output exactly:
  "No concrete bug can be established from the available code context."

SIMILAR_CODE:
- Only compare code that appears in the retrieved context.
- Compare observable things only: inputs, outputs, decorators, control flow, or data structures.
- Do NOT claim that functions interact, call each other, or form a pipeline unless the retrieved code explicitly shows that relationship.
- Do not add unrelated architecture commentary.

FUNCTION_SEARCH:
- Return only functions literally present in the retrieved context.
- Include function name, file path, line range, and one-sentence relevance.

CLASS_SEARCH:
- Return ONLY class declarations literally present in the retrieved context.
- Before naming a class, verify that the context contains the literal declaration:
  "class <Name>"
- A variable, dictionary, function name, or concept is NEVER evidence that a class exists.
- NEVER infer a class name.
- If no class declaration is present, output exactly:
  "No relevant classes were found in the available code context."
- Do not explain hypothetical classes.

GENERAL:
- Answer directly from retrieved code.

EXPLAIN:
- Explain the retrieved code step by step.
"""

    TASK_PREFIXES = {
        "general": "Answer the following question about the codebase:",
        "explain": "Explain the following code or concept:",
        "find_bugs": "Identify potential bugs, errors, or issues in the following context:",
        "similar_code": "Find and describe code patterns similar to what is being asked:",
        "function_search": "Find and explain the function(s) related to the following:",
        "class_search": "Find and explain the class(es) related to the following:",
    }

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        task_type: str = "general",
        max_context_chars: int = 8000,
    ) -> str:
        """
        Build the user prompt with injected retrieved context.

        Args:
            question: The user's question.
            retrieved_chunks: Ranked and reranked code chunks.
            task_type: One of the keys in TASK_PREFIXES.
            max_context_chars: Maximum characters of context to inject.

        Returns:
            The formatted prompt string.
        """
        prefix = self.TASK_PREFIXES.get(task_type, self.TASK_PREFIXES["general"])

        context_blocks: List[str] = []
        total_chars = 0

        for i, chunk in enumerate(retrieved_chunks, start=1):
            block = self._format_chunk(i, chunk)
            if total_chars + len(block) > max_context_chars:
                logger.debug(
                    "Context limit reached at chunk %d/%d", i, len(retrieved_chunks)
                )
                break
            context_blocks.append(block)
            total_chars += len(block)

        context_section = "\n\n".join(context_blocks)

        prompt = f"""{prefix}

**Question:** {question}

---

## CODE CONTEXT

{context_section}

---"""

        return prompt

    def build_chat_prompt(
        self,
        question: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_history: List[Dict[str, str]],
        task_type: str = "general",
    ) -> List[Dict[str, str]]:
        """
        Build a full messages list for multi-turn conversation.

        Args:
            question: Current user question.
            retrieved_chunks: Retrieved chunks for this turn.
            conversation_history: Previous messages [{"role": ..., "content": ...}].
            task_type: Task type key.

        Returns:
            Messages list ready for OllamaClient.generate_with_history().
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(conversation_history)

        user_prompt = self.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            task_type=task_type,
        )
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _format_chunk(self, index: int, chunk: RetrievedChunk) -> str:
        """Format a single code chunk into the context block."""
        header_parts = [f"**[{index}] {chunk.chunk_type.upper()}**"]

        if chunk.function_name:
            header_parts.append(f"`{chunk.function_name}`")
        elif chunk.class_name:
            header_parts.append(f"`{chunk.class_name}`")

        header_parts.append(f"| 📄 `{chunk.file_path}` (L{chunk.start_line}–{chunk.end_line})")
        header_parts.append(f"| 🔤 {chunk.language}")

        if chunk.rerank_score:
            header_parts.append(f"| Score: {chunk.rerank_score:.3f}")

        header = " ".join(header_parts)
        lang = chunk.language or "code"

        return f"""{header}
```{lang}
{chunk.raw_code}
```"""


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestrates Vector + BM25 retrieval with RRF fusion and Cross Encoder reranking.

    This is the core retrieval engine. It:
    1. Runs vector search and BM25 concurrently.
    2. Merges results via Reciprocal Rank Fusion.
    3. Optionally reranks via Cross Encoder.
    4. Returns the final top-K RetrievedChunk list.
    """

    # Generic English filler/question words to strip when pulling "concept"
    # terms out of a user's question (see _extract_concept_terms). Contains
    # no repository-specific names — every concept term that survives this
    # filter comes from what the user actually typed (e.g. "bm25", "rrf",
    # "cross", "encoder", "chromadb", "indexing").
    _QUESTION_STOPWORDS = frozenset({
        "how", "does", "do", "did", "is", "are", "was", "were", "the", "a",
        "an", "in", "on", "of", "for", "to", "this", "that", "it", "and",
        "or", "work", "works", "working", "implement", "implemented",
        "implementation", "where", "used", "use", "uses", "using",
        "repository", "repo", "project", "code", "here", "what", "why",
        "when", "which", "explain", "describe", "detail", "details",
        "about", "with", "system", "you", "your", "please", "tell", "me",
    })

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        config: Optional[RetrievalConfig] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.config = config or RetrievalConfig()
        self._reranker = reranker

        # RRF fusion weights applied to the vector-search and BM25 ranked
        # lists respectively (see reciprocal_rank_fusion). These live here
        # rather than in RetrievalConfig (config.py is intentionally left
        # untouched) — pass explicit values at construction time to
        # override the defaults.
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        if self.config.use_reranker and self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.config.cross_encoder_model,
                device=self.config.reranker_device,
            )

    def retrieve(
        self,
        query: str,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
        diversify: bool = False,
        boost_query_symbols: bool = False,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Execute the full hybrid retrieval pipeline.

        Args:
            query: The search query string.
            language: Optional language filter (e.g., "python").
            chunk_type: Optional chunk type filter (e.g., "function").
            file_path: Optional file path filter.
            class_name: Optional class name filter.
            diversify: When False (default), behavior is unchanged from
                before this flag existed — the cross encoder reranks
                straight down to final_top_k, one chunk's neighbors can
                dominate the result set, exactly as today. When True,
                the cross encoder instead reranks the FULL rerank_candidates
                pool (still the same reranker, same scoring, same
                threshold — just asked to return more of its own output),
                and a diversity-aware selection then picks final_top_k from
                that larger already-reranked pool. Intended for
                architecture/project-structure queries, where 5 near-
                duplicate chunks from one class are worse than 5 chunks
                spanning the actual project structure.
            boost_query_symbols: When True, before the cross encoder ever
                runs, generic concept terms extracted from `query` itself
                (see _extract_concept_terms) are matched against
                function_name/class_name/chunk_type/code of every chunk
                already present in the fused vector+BM25 result set (no
                new retrieval call). Any match not already inside the
                rerank_candidates window is promoted into it, displacing
                the weakest non-matching candidate, so a chunk whose RRF
                rank alone would have excluded it from reranking still
                gets a chance to be judged by the cross encoder. Vector
                search, BM25, and RRF scoring themselves are untouched —
                this only affects which chunks reach the reranker.

        Returns:
            Tuple of (retrieved_chunks, timing_breakdown).
            timing_breakdown keys: vector_ms, bm25_ms, rrf_ms, rerank_ms, total_ms.
        """
        timings: Dict[str, float] = {}
        where_filter = self.vector_store.metadata_filter(
            language=language,
            chunk_type=chunk_type,
            file_path=file_path,
            class_name=class_name,
        )

        # --- Step 1: Concurrent vector + BM25 search ---
        t_start = time.monotonic()
        vector_results: List[Dict] = []
        bm25_results: List[Dict] = []

        if self.config.concurrent_retrieval:
            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(
                    self.vector_store.vector_search,
                    query, self.config.top_k_vector, where_filter,
                )
                bm25_future = executor.submit(
                    self.bm25_index.search,
                    query, self.config.top_k_bm25,
                    language, chunk_type,
                )
                vector_results = vector_future.result()
                bm25_results = bm25_future.result()
        else:
            t0 = time.monotonic()
            vector_results = self.vector_store.vector_search(
                query, self.config.top_k_vector, where_filter
            )
            timings["vector_ms"] = (time.monotonic() - t0) * 1000

            t0 = time.monotonic()
            bm25_results = self.bm25_index.search(
                query, self.config.top_k_bm25, language, chunk_type
            )
            timings["bm25_ms"] = (time.monotonic() - t0) * 1000

        timings.setdefault("vector_ms", 0.0)
        timings.setdefault("bm25_ms", 0.0)

        logger.debug(
            "Retrieved %d vector + %d BM25 candidates",
            len(vector_results), len(bm25_results),
        )

        # --- Step 2: Reciprocal Rank Fusion (vector/BM25-weighted) ---
        t0 = time.monotonic()
        fused = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=self.config.rrf_k,
            weights=[self.vector_weight, self.bm25_weight],
        )
        timings["rrf_ms"] = (time.monotonic() - t0) * 1000

        # Take top rerank_candidates for the cross encoder
        candidates = fused[:self.config.rerank_candidates]

        # --- Optional: promote query-symbol matches into the pool ---
        # Pure in-memory lookup over `fused` (already fetched above) — no
        # second vector_search/bm25_index.search call. Only changes WHICH
        # chunks the cross encoder sees; RRF ranking/scores are untouched.
        if boost_query_symbols:
            concept_terms = self._extract_concept_terms(query)
            if concept_terms:
                candidates = self._promote_symbol_matches(
                    fused, candidates, concept_terms, self.config.rerank_candidates
                )

        # --- Step 3: Cross Encoder Reranking ---
        # diversify=False (default): top_n=final_top_k, identical to before
        # this flag existed. diversify=True: top_n=rerank_candidates — the
        # SAME reranker, same model, same score_threshold, just returning
        # more of the candidates it already scored, so diversity selection
        # below has a real pool to choose from instead of only ever seeing
        # whatever final_top_k the reranker would have kept anyway.
        rerank_pool_size = self.config.rerank_candidates if diversify else self.config.final_top_k
        t0 = time.monotonic()
        if self.config.use_reranker and self._reranker and candidates:
            reranked = self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=rerank_pool_size,
                score_threshold=self.config.rerank_score_threshold,
            )
        else:
            reranked = candidates[:rerank_pool_size]
            for i, item in enumerate(reranked):
                item["rerank_score"] = item.get("rrf_score", 0.0)

        timings["rerank_ms"] = (time.monotonic() - t0) * 1000
        timings["total_ms"] = (time.monotonic() - t_start) * 1000

        # --- Step 4: Build scores index for annotation ---
        vector_scores = {r["chunk_id"]: (r.get("score", 0.0), r.get("rank", 0)) for r in vector_results}
        bm25_scores = {r["chunk_id"]: (r.get("score", 0.0), r.get("rank", 0)) for r in bm25_results}
        rrf_scores = {r["chunk_id"]: r.get("rrf_score", 0.0) for r in fused}

        # --- Step 5: Assemble RetrievedChunk objects ---
        chunks: List[RetrievedChunk] = []
        for item in reranked:
            cid = item["chunk_id"]
            v_score, v_rank = vector_scores.get(cid, (0.0, 0))
            b_score, b_rank = bm25_scores.get(cid, (0.0, 0))
            rc = RetrievedChunk(
                chunk_id=cid,
                document=item["document"],
                metadata=item["metadata"],
                vector_score=v_score,
                bm25_score=b_score,
                vector_rank=v_rank,
                bm25_rank=b_rank,
                rrf_score=rrf_scores.get(cid, 0.0),
                rerank_score=item.get("rerank_score", 0.0),
                final_rank=item.get("rank", 0),
            )
            chunks.append(rc)

        # --- Step 5.5: prefer implementation chunks over test chunks that ---
        # --- match the same concept (implementation questions only) ---
        # Cross-encoder scores are left untouched; this only re-orders the
        # already-scored `chunks` list before the final_top_k cutoff below,
        # so a real implementation chunk that reached the reranker (via
        # boost_query_symbols promotion) isn't pushed out of the result by
        # a more "quotable" test/docstring chunk describing the same
        # concept. No-op unless boost_query_symbols was used.
        if boost_query_symbols and concept_terms:
            chunks = self._prefer_implementation_over_tests(chunks, concept_terms)

        # Keep a handle to the full reranked pool (post Step 5.5, pre any
        # truncation below) — Step 6.5 needs it to recover an
        # implementation chunk that scored well but still fell outside
        # final_top_k, without re-running vector/BM25/RRF/cross-encoder.
        reranked_pool = chunks

        # --- Step 6: Diversity-aware selection (architecture queries only) ---
        # Runs AFTER reranking, using metadata already on each RetrievedChunk
        # (file_path/class_name/chunk_type) — no extra retrieval call, no
        # hard-coded filenames. No-op when diversify=False.
        if diversify:
            chunks = self._select_diverse(chunks, limit=self.config.final_top_k)

        # --- Step 6.5: implementation-coverage safeguard (implementation ---
        # --- questions only) ---
        # Cross-encoder scores/order are never touched here. For each
        # concept term in the query, this finds the single strongest
        # (highest rerank_score) non-test implementation chunk anywhere in
        # the reranked pool. If it isn't already in the final selection
        # (e.g. it was outscored by several non-matching or test chunks
        # and truncated away), it displaces the weakest chunk in the
        # current selection that isn't itself protecting a different
        # concept — never dropping below final_top_k, never forcing in
        # implementation chunks that don't match anything asked about.
        if boost_query_symbols and concept_terms:
            chunks = self._ensure_implementation_coverage(
                reranked_pool, chunks, concept_terms, self.config.final_top_k
            )

        return chunks, timings

    @staticmethod
    def _select_diverse(chunks: List["RetrievedChunk"], limit: int) -> List["RetrievedChunk"]:
        """
        Greedy diversity selection over an already-reranked pool.

        `chunks` is assumed sorted by relevance (rerank_score) descending,
        exactly as the cross encoder / RRF-fallback already produced it —
        this function only chooses WHICH of those already-ranked chunks to
        keep, it does not re-score anything, so relevance ordering is
        preserved as much as diversity allows: the single most relevant
        chunk per (file_path, class_name, chunk_type) combination is taken
        first, in relevance order, before any duplicate combination is
        allowed in. If that doesn't fill `limit` (e.g. a genuinely narrow
        result set with few distinct files/classes), the next-most-relevant
        leftover chunks pad the rest, so this never returns fewer chunks
        than plain reranking would have for the same pool.
        """
        if len(chunks) <= limit:
            return chunks

        selected: List["RetrievedChunk"] = []
        leftover: List["RetrievedChunk"] = []
        seen_keys = set()

        for c in chunks:
            key = (c.file_path, c.class_name, c.chunk_type)
            if key not in seen_keys:
                seen_keys.add(key)
                selected.append(c)
            else:
                leftover.append(c)
            if len(selected) == limit:
                break

        if len(selected) < limit:
            selected.extend(leftover[: limit - len(selected)])

        # Renumber final_rank to reflect the presented order post-selection
        # (raw reranker ranks, e.g. 1/3/7, would otherwise leak through as
        # non-contiguous once duplicates are skipped).
        for i, c in enumerate(selected, start=1):
            c.final_rank = i

        return selected

    @classmethod
    def _extract_concept_terms(cls, query: str) -> List[str]:
        """
        Pull generic "concept" terms out of the user's own question text —
        e.g. "bm25", "rrf", "cross", "encoder", "chromadb", "indexing" —
        by stripping common English question/filler words. No repository-
        specific names are hard-coded anywhere in this method; every term
        that survives comes directly from what the user typed.
        """
        tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
        return [t for t in tokens if len(t) >= 3 and t not in cls._QUESTION_STOPWORDS]

    @staticmethod
    def _promote_symbol_matches(
        fused: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        concept_terms: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Ensure chunks whose function_name/class_name/chunk_type or raw code
        contain one of the query's concept terms are present in the
        reranker's candidate pool, even if their RRF rank alone put them
        past the rerank_candidates cutoff.

        Operates entirely on `fused` (the full, already-fetched vector+BM25
        RRF result set) — no new retrieval call. RRF scores/order are not
        modified: matches are appended in their existing RRF order, and
        only the weakest non-matching candidates (the tail of the
        RRF-sorted `candidates` list) are displaced to keep the pool at
        `limit`, so the cross encoder still performs the final ranking
        exactly as before, just over a pool with better coverage of the
        concepts the user actually asked about.
        """
        candidate_ids = {c["chunk_id"] for c in candidates}
        matches: List[Dict[str, Any]] = []
        for item in fused:
            if item["chunk_id"] in candidate_ids:
                continue
            metadata = item.get("metadata") or {}
            haystack = " ".join(
                str(metadata.get(field, ""))
                for field in ("function_name", "class_name", "chunk_type")
            )
            haystack += " " + str(item.get("document", ""))
            haystack = haystack.lower()
            if any(term in haystack for term in concept_terms):
                matches.append(item)

        if not matches:
            return candidates

        promoted = list(candidates)
        replace_idx = len(promoted) - 1  # weakest original candidate, RRF-sorted tail
        for item in matches:
            if len(promoted) < limit:
                promoted.append(item)
            elif replace_idx >= 0:
                promoted[replace_idx] = item
                replace_idx -= 1
            else:
                break  # every slot already holds a promoted match

        return promoted

    @staticmethod
    def _is_test_chunk(metadata: Dict[str, Any]) -> bool:
        """
        Generic test-chunk detection from existing metadata only — no
        repository-specific file/class names. A chunk counts as a test
        chunk if its file path has a test-ish path segment/filename, or
        its class/function name is a test name (standard pytest/unittest
        conventions: `test_*.py`, `*_test.py`, `tests/` directory,
        `Test*` classes, `test_*` functions).
        """
        file_path = str(metadata.get("file_path", "")).lower()
        class_name = str(metadata.get("class_name", "")).lower()
        function_name = str(metadata.get("function_name", "")).lower()

        path_parts = re.split(r"[\\/]", file_path)
        if any(part in ("test", "tests") for part in path_parts):
            return True

        fname = path_parts[-1] if path_parts else ""
        if fname.startswith("test_") or fname.endswith("_test.py") or fname.endswith("_test"):
            return True

        if class_name.startswith("test"):
            return True
        if function_name.startswith("test_") or function_name.startswith("test"):
            return True

        return False

    @classmethod
    def _prefer_implementation_over_tests(
        cls,
        chunks: List["RetrievedChunk"],
        concept_terms: List[str],
    ) -> List["RetrievedChunk"]:
        """
        Stable re-order of an already cross-encoder-scored `chunks` list:
        for any concept term that at least one implementation (non-test)
        chunk in the pool matches, push test chunks matching that SAME
        term below the rest of the pool. Cross-encoder scores are not
        touched — this only changes selection order, and only for the
        overlapping concept(s); a test chunk that doesn't share a matched
        concept with any implementation chunk keeps its reranked position
        exactly as-is, so tests remain fully usable for queries where no
        implementation chunk covers the same concept.
        """
        if not concept_terms or not chunks:
            return chunks

        def matched_terms(c: "RetrievedChunk") -> set:
            haystack = " ".join(
                [c.function_name, c.class_name, c.chunk_type, c.document]
            ).lower()
            return {t for t in concept_terms if t in haystack}

        impl_matched_terms: set = set()
        for c in chunks:
            if not cls._is_test_chunk(c.metadata):
                impl_matched_terms |= matched_terms(c)

        if not impl_matched_terms:
            # No implementation chunk in this pool covers any concept the
            # question asked about — nothing to prefer over tests here.
            return chunks

        def sort_key(indexed: Tuple[int, "RetrievedChunk"]) -> Tuple[int, int]:
            idx, c = indexed
            demote = cls._is_test_chunk(c.metadata) and bool(
                matched_terms(c) & impl_matched_terms
            )
            return (1 if demote else 0, idx)

        ordered = [c for _, c in sorted(enumerate(chunks), key=sort_key)]
        for i, c in enumerate(ordered, start=1):
            c.final_rank = i
        return ordered

    @classmethod
    def _ensure_implementation_coverage(
        cls,
        pool: List["RetrievedChunk"],
        selected: List["RetrievedChunk"],
        concept_terms: List[str],
        limit: int,
    ) -> List["RetrievedChunk"]:
        """
        Post-truncation safeguard: for each query concept term, find the
        single highest rerank_score non-test implementation chunk anywhere
        in the full reranked `pool`. If it didn't survive into `selected`
        (e.g. it was crowded out by wrapper/demo/class-level chunks that
        also happened to mention the concept), swap it in for the weakest
        chunk currently in `selected` that isn't itself protecting a
        different concept's match. Cross-encoder scores are read-only
        here — this only changes which already-scored chunks make the
        final cut, and only for implementation-style queries.
        """
        if not concept_terms or not pool:
            return selected

        def matched_terms(c: "RetrievedChunk") -> set:
            haystack = " ".join(
                [c.function_name, c.class_name, c.chunk_type, c.document]
            ).lower()
            return {t for t in concept_terms if t in haystack}

        # Strongest real-implementation (non-test) chunk per concept term.
        best_per_term: Dict[str, "RetrievedChunk"] = {}
        for c in pool:
            if cls._is_test_chunk(c.metadata):
                continue
            for term in matched_terms(c):
                current = best_per_term.get(term)
                if current is None or c.rerank_score > current.rerank_score:
                    best_per_term[term] = c

        required = {c.chunk_id: c for c in best_per_term.values()}
        if not required:
            return selected

        selected_ids = {c.chunk_id for c in selected}
        missing = [c for cid, c in required.items() if cid not in selected_ids]
        if not missing:
            return selected

        result = list(selected)
        missing.sort(key=lambda c: c.rerank_score, reverse=True)

        for m in missing:
            if len(result) < limit:
                result.append(m)
                continue
            # Displace the weakest currently-selected chunk that isn't
            # itself a required best-per-concept match — never evict one
            # guaranteed concept match to make room for another.
            weakest_idx = None
            weakest_score = None
            for i, c in enumerate(result):
                if c.chunk_id in required:
                    continue
                if weakest_score is None or c.rerank_score < weakest_score:
                    weakest_score = c.rerank_score
                    weakest_idx = i
            if weakest_idx is None:
                # Every current slot already protects a concept match —
                # no safe room to add another without losing coverage
                # elsewhere.
                continue
            result[weakest_idx] = m

        # Re-sort by the cross encoder's own rerank_score (untouched) and
        # renumber presented rank — same ordering rule _select_diverse
        # already uses after its own metadata-based selection.
        result.sort(key=lambda c: c.rerank_score, reverse=True)
        for i, c in enumerate(result, start=1):
            c.final_rank = i
        return result


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    Composes HybridRetriever + PromptBuilder + OllamaClient into a single
    coherent interface used by the CLI and API.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        from config import DEFAULT_CONFIG
        self.config = config or DEFAULT_CONFIG

        # Resolve the active repository exactly once, here. Every other
        # method on this pipeline (index_repository, query, search, reset,
        # import_data) works through self._vector_store / self._bm25_index,
        # both already scoped to whichever repository was active at
        # construction time — nothing downstream re-resolves or re-derives
        # this, so a pipeline instance stays consistently scoped to one
        # repository for its whole lifetime even if the registry's active
        # pointer changes later (a fresh RAGPipeline() picks up the change).
        from repository_manager import RepositoryManager
        self.repository: Optional[dict] = RepositoryManager().get_active()

        if self.repository:
            collection_name = self.repository["collection"]
            bm25_dir = self.repository["bm25_dir"]
            logger.info(
                "RAGPipeline scoped to repository '%s' (collection='%s', bm25_dir='%s')",
                self.repository["name"], collection_name, bm25_dir,
            )
        else:
            # No repository registered/active — preserve pre-multi-repo
            # single-repository behavior exactly: static collection_name /
            # bm25_index_dir from VectorStoreConfig, same as before this change.
            collection_name = self.config.vector_store.collection_name
            bm25_dir = self.config.vector_store.bm25_index_dir

        self._vector_store = VectorStore(self.config.vector_store, collection_name=collection_name)
        # Remembered so reset() (and anything else that needs it) reuses
        # the SAME resolved directory rather than re-deriving it from
        # self.config.vector_store.bm25_index_dir, which would silently
        # ignore repository scoping.
        self._bm25_dir = bm25_dir
        self._bm25_index = BM25Index(self._bm25_dir)
        self._retriever: Optional[HybridRetriever] = None
        self._llm = OllamaClient(self.config.llm)
        self._prompt_builder = PromptBuilder()

        self._ensure_bm25_loaded()

    def _ensure_bm25_loaded(self) -> None:
        """Load BM25 index from disk if available; otherwise it will be built on first index."""
        if not self._bm25_index.load():
            if self._vector_store.count > 0:
                logger.info("Rebuilding BM25 index from ChromaDB collection...")
                items = self._vector_store.get_all_documents()
                self._bm25_index.build(items)
            else:
                logger.info("No indexed data found. Run `privarepo index <repo>` first.")

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                vector_store=self._vector_store,
                bm25_index=self._bm25_index,
                config=self.config.retrieval,
            )
        return self._retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_repository(
        self,
        repo_path: str,
        include_extensions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Full indexing pipeline: parse → chunk → embed → store → build BM25.

        Returns a summary dict with chunk counts, timing, etc.
        """
        from tree_sitter_chunker import TreeSitterChunker

        chunker = TreeSitterChunker(self.config.chunker)

        t_parse_start = time.monotonic()
        chunks = chunker.chunk_repository(
            repo_path=repo_path,
            include_extensions=include_extensions,
        )
        parse_time = time.monotonic() - t_parse_start

        if not chunks:
            return {"status": "warning", "message": "No chunks extracted from repository."}

        t_index_start = time.monotonic()
        n_indexed = self._vector_store.add_chunks(chunks, show_progress=show_progress)
        index_time = time.monotonic() - t_index_start

        # Rebuild BM25 from updated collection
        t_bm25_start = time.monotonic()
        items = self._vector_store.get_all_documents()
        self._bm25_index.build(items)
        bm25_time = time.monotonic() - t_bm25_start

        # Invalidate cached retriever so it gets a fresh BM25 reference
        self._retriever = None

        return {
            "status": "success",
            "chunks_extracted": len(chunks),
            "chunks_indexed": n_indexed,
            "parse_time_seconds": round(parse_time, 2),
            "index_time_seconds": round(index_time, 2),
            "bm25_build_time_seconds": round(bm25_time, 2),
            "total_time_seconds": round(parse_time + index_time + bm25_time, 2),
            "collection_total": self._vector_store.count,
        }

    def query(
        self,
        question: str,
        task_type: str = "general",
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
        file_path: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute a full RAG query: retrieve → build prompt → generate.

        Args:
            question: The user's question.
            task_type: One of: general, explain, find_bugs, similar_code, function_search, class_search.
            language: Filter results by programming language.
            chunk_type: Filter by chunk type (function, class, method, etc.).
            file_path: Filter by source file.
            class_name: Filter by class name.

        Returns:
            RAGResponse with answer, reasoning, citations, and timing.
        """
        retriever = self._get_retriever()

        t_total_start = time.monotonic()

        # Retrieval
        t_ret_start = time.monotonic()

        architecture_tasks = {
            "architecture",
            "project_structure",
            "structure",
            "design",
        }

        is_architecture_question = (
            task_type in architecture_tasks
            or any(
                phrase in question.lower()
                for phrase in (
                    "project structure",
                    "structure of this project",
                    "architecture",
                    "overall architecture",
                    "how is the project structured",
                    "how does the project fit together",
                    "how is the system structured",
                    "how do the components interact",
                )
            )
        )

        # "How does X work" / "where is X implemented" style questions ask
        # about a specific mechanism's actual code, not whatever chunk is
        # merely lexically/semantically closest to the plain-English
        # phrasing of the question. A thin caller/wrapper (e.g. a one-line
        # `search()` method) or a prose docstring that happens to name the
        # concept can out-rank the real implementation body in vector/BM25
        # scoring. Two compounding effects follow: (1) with diversify=False
        # the reranker only ever looks at a small final_top_k window, so
        # that wrapper/docstring chunk can take every slot; and (2) even
        # with diversify=True, the real implementation chunk may never
        # reach reranking at all if its RRF rank alone put it past the
        # rerank_candidates cutoff — diversify only reorders what already
        # got that far. Detected only from generic phrasing already in the
        # question (no repo-specific class/file names), and handled by (a)
        # reusing the SAME diversify=True path the architecture branch
        # above already uses — no second retrieval call, just a wider
        # reranked pool deduplicated by (file_path, class_name, chunk_type)
        # — and (b) boost_query_symbols=True, which promotes chunks whose
        # metadata/code match a concept term from the question (e.g.
        # "bm25", "rrf") into that pool BEFORE the rerank_candidates cutoff,
        # again with no second retrieval call — see
        # HybridRetriever._promote_symbol_matches.
        implementation_tasks = {
            "algorithm",
            "implementation",
        }

        is_implementation_question = (
            task_type in implementation_tasks
            or (
                ("how does" in question.lower() and "work" in question.lower())
                or ("how is" in question.lower() and "implement" in question.lower())
                or ("where is" in question.lower() and "implement" in question.lower())
                or "implementation of" in question.lower()
            )
        )

        if is_architecture_question and chunk_type is None:
            chunks, _ = retriever.retrieve(
                query=question,
                language=language,
                chunk_type="class",
                file_path=file_path,
                class_name=class_name,
                diversify=True,
            )

            # If class-level retrieval is too narrow, fall back to normal hybrid retrieval.
            # NOTE: this runs retrieval a second time when it triggers, roughly doubling
            # retrieval_time for this request — a real latency cost, not just a fallback
            # in name only. Worth confirming via `benchmark` that it's worth paying.
            if not chunks:
                chunks, _ = retriever.retrieve(
                    query=question,
                    language=language,
                    chunk_type=None,
                    file_path=file_path,
                    class_name=class_name,
                    diversify=True,
                )
        elif is_implementation_question and chunk_type is None:
            chunks, _ = retriever.retrieve(
                query=question,
                language=language,
                chunk_type=chunk_type,
                file_path=file_path,
                class_name=class_name,
                diversify=True,
                boost_query_symbols=True,
            )
        else:
            chunks, _ = retriever.retrieve(
                query=question,
                language=language,
                chunk_type=chunk_type,
                file_path=file_path,
                class_name=class_name,
            )

        retrieval_time = time.monotonic() - t_ret_start

        if not chunks:
            return RAGResponse(
                question=question,
                answer="No relevant code was found in the indexed repository for this query.",
                reasoning="The retrieval pipeline returned zero results.",
                referenced_files=[],
                functions_used=[],
                retrieved_chunks=[],
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=time.monotonic() - t_total_start,
                model=self.config.llm.model,
                filters_applied={"language": language or "", "chunk_type": chunk_type or ""},
            )

        # --- Deterministic class_search safeguard ---
        # The 1.5B LLM is not reliable enough to obey "don't hallucinate a
        # class name" purely via prompt instructions. So for class_search we
        # check the retrieved chunks' own metadata BEFORE generation: a
        # chunk only carries class_name / chunk_type=="class" if the
        # Tree-sitter chunker (tree_sitter_chunker.py) actually parsed a
        # real `class <Name>` declaration for it — this is not an inference,
        # it's the AST fact already attached to the chunk. If none of the
        # retrieved chunks have that evidence, there is no class to report,
        # so we skip the LLM entirely and return the fixed answer — this is
        # the only way to guarantee zero hallucination for this task type.
        if task_type == "class_search":
            has_class_evidence = any(
                c.chunk_type == "class" or c.class_name for c in chunks
            )
            if not has_class_evidence:
                return RAGResponse(
                    question=question,
                    answer="No relevant classes were found in the available code context.",
                    reasoning="No class declaration was present in the retrieved context.",
                    referenced_files=list({c.file_path for c in chunks}),
                    functions_used=[c.function_name for c in chunks if c.function_name],
                    retrieved_chunks=chunks,
                    retrieval_time=retrieval_time,
                    generation_time=0.0,
                    total_time=time.monotonic() - t_total_start,
                    model=self.config.llm.model,
                    filters_applied={
                        "language": language or "",
                        "chunk_type": chunk_type or "",
                        "file_path": file_path or "",
                        "class_name": class_name or "",
                    },
                )

        # Build prompt
        prompt = self._prompt_builder.build_prompt(
            question=question,
            retrieved_chunks=chunks,
            task_type=task_type,
        )

        # Generate
        t_gen_start = time.monotonic()
        llm_response = self._llm.generate(
            prompt=prompt,
            system_prompt=PromptBuilder.SYSTEM_PROMPT,
        )
        generation_time = time.monotonic() - t_gen_start

        total_time = time.monotonic() - t_total_start

        return RAGResponse(
            question=question,
            answer=llm_response.answer,
            reasoning=llm_response.reasoning,
            referenced_files=llm_response.referenced_files or list({c.file_path for c in chunks}),
            functions_used=llm_response.functions_used or [c.function_name for c in chunks if c.function_name],
            retrieved_chunks=chunks,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            model=llm_response.model,
            filters_applied={
                "language": language or "",
                "chunk_type": chunk_type or "",
                "file_path": file_path or "",
                "class_name": class_name or "",
            },
        )

    def search(
        self,
        query: str,
        language: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], Dict[str, float]]:
        """
        Raw hybrid search without LLM generation.
        Returns (chunks, timing_breakdown).
        """
        retriever = self._get_retriever()
        return retriever.retrieve(
            query=query,
            language=language,
            chunk_type=chunk_type,
        )

    def get_stats(self) -> Dict[str, Any]:
        stats = self._vector_store.get_stats()
        stats["bm25_index_size"] = self._bm25_index.size
        stats["bm25_index_built"] = self._bm25_index.is_built
        return stats

    def reset(self) -> None:
        self._vector_store.reset()
        # Clear BM25 index files — reuse the same resolved bm25_dir this
        # pipeline instance was constructed with, not a fresh re-derivation
        # from config, so reset() stays correct for whichever repository
        # (or the no-repository single-repo fallback) is actually active.
        bm25_dir = Path(self._bm25_dir)
        for f in bm25_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
        self._bm25_index = BM25Index(self._bm25_dir)
        self._retriever = None
        logger.info("Pipeline reset complete.")

    def export(self, output_path: str) -> int:
        return self._vector_store.export_collection(output_path)

    def import_data(self, input_path: str, reset_first: bool = False) -> int:
        n = self._vector_store.import_collection(input_path, reset_first=reset_first)
        # Rebuild BM25 after import
        items = self._vector_store.get_all_documents()
        self._bm25_index.build(items)
        self._retriever = None
        return n

    @property
    def llm(self) -> OllamaClient:
        return self._llm

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    @property
    def bm25_index(self) -> BM25Index:
        return self._bm25_index