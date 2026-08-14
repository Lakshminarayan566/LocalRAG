"""
evaluator.py — Retrieval & Generation Evaluation Suite

Implements RAGAS-style metrics using the local Ollama LLM as judge
(no external API calls). Measures:

Retrieval:
  - Precision@K, Recall@K, MRR, Hit Rate

Generation:
  - Faithfulness (NLI-like: is the answer supported by context?)
  - Answer Relevancy (does the answer address the question?)
  - Context Precision (is retrieved context relevant?)
  - Context Recall (does context cover the expected answer?)

Latency:
  - P50, P95, P99 for retrieval, generation, and total pipeline

Design decisions:
  - LLM-as-judge approach for faithfulness/relevancy avoids needing
    a separate NLI model and produces human-like quality scores.
  - Metrics are computed per-query and aggregated with mean/std.
  - Results are saved as structured JSON for CI/CD integration.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from config import AppConfig, EvalConfig
from llm_interface import OllamaClient
from rag_pipeline import RAGPipeline, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Data Models
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""

    query_id: str
    question: str
    expected_answer: Optional[str] = None
    relevant_chunk_ids: List[str] = field(default_factory=list)   # Ground truth chunk IDs
    relevant_files: List[str] = field(default_factory=list)        # Ground truth files
    relevant_functions: List[str] = field(default_factory=list)    # Ground truth functions
    task_type: str = "general"
    language_filter: Optional[str] = None


@dataclass
class QueryRetrievalMetrics:
    """Per-query retrieval evaluation metrics."""

    query_id: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float  # 1/rank_of_first_relevant
    hit: bool               # True if any relevant chunk appears in top-K
    k: int
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class QueryGenerationMetrics:
    """Per-query generation evaluation metrics."""

    query_id: str
    faithfulness: float       # [0, 1] — is the answer supported by context?
    answer_relevancy: float   # [0, 1] — does the answer address the question?
    context_precision: float  # [0, 1] — fraction of context chunks that are relevant
    context_recall: float     # [0, 1] — fraction of ground truth covered by context


@dataclass
class RetrievalBenchmarkResult:
    """Aggregated retrieval benchmark across all queries."""

    precision_at_k: float
    recall_at_k: float
    mrr: float               # Mean Reciprocal Rank
    hit_rate: float
    k: int
    n_queries: int
    precision_std: float = 0.0
    recall_std: float = 0.0
    per_query: List[QueryRetrievalMetrics] = field(default_factory=list)


@dataclass
class GenerationBenchmarkResult:
    """Aggregated generation benchmark across all queries."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_queries: int
    faithfulness_std: float = 0.0
    answer_relevancy_std: float = 0.0
    per_query: List[QueryGenerationMetrics] = field(default_factory=list)
    # Queries that raised an exception during generation/eval and were
    # excluded from the metrics above (rather than silently defaulted in).
    n_failed: int = 0
    failed_query_ids: List[str] = field(default_factory=list)
    # LLM-judge parsing diagnostics — how often score_* had to fall back
    # to the neutral 0.5 because the judge's response couldn't be parsed.
    judge_total_evaluations: int = 0
    judge_fallback_count: int = 0
    judge_fallback_rate: float = 0.0


@dataclass
class LatencyBenchmarkResult:
    """Statistical latency profile from N benchmark runs."""

    n_runs: int
    retrieval_p50_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float
    generation_p50_ms: float
    generation_p95_ms: float
    generation_p99_ms: float
    total_p50_ms: float
    total_p95_ms: float
    total_p99_ms: float
    memory_usage_mb: float
    collection_size: int
    raw_retrieval_ms: List[float] = field(default_factory=list)
    raw_generation_ms: List[float] = field(default_factory=list)
    raw_total_ms: List[float] = field(default_factory=list)
    # Runs that raised an exception and were excluded from the
    # percentiles above (rather than recorded as a misleading 0.0).
    n_failed: int = 0


@dataclass
class FullBenchmarkReport:
    """Complete benchmark report combining all metric categories."""

    timestamp: str
    system_info: Dict[str, Any]
    collection_stats: Dict[str, Any]
    retrieval: Optional[RetrievalBenchmarkResult] = None
    generation: Optional[GenerationBenchmarkResult] = None
    latency: Optional[LatencyBenchmarkResult] = None


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses the local Ollama LLM to score faithfulness and relevancy.

    Faithfulness: Is every statement in the answer supported by the context?
    Relevancy: Does the answer directly and completely address the question?
    Context Precision: Is each retrieved context chunk actually relevant?
    Context Recall: Does the context cover all key information needed for the answer?
    """

    FAITHFULNESS_PROMPT = """\
You are an objective judge evaluating the faithfulness of an AI-generated answer.

QUESTION: {question}

CONTEXT (retrieved code chunks):
{context}

ANSWER:
{answer}

TASK: Evaluate whether every statement in the ANSWER is directly supported by the CONTEXT.
Score from 0.0 to 1.0:
  1.0 = Fully faithful, every claim in the answer is grounded in the context
  0.5 = Partially faithful, some claims are supported but others are not
  0.0 = Not faithful, the answer contains hallucinations or unsupported claims

Respond with ONLY a JSON object in this format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence justification>"}}"""

    RELEVANCY_PROMPT = """\
You are an objective judge evaluating the relevancy of an AI-generated answer.

QUESTION: {question}

ANSWER:
{answer}

TASK: Evaluate whether the ANSWER directly and completely addresses the QUESTION.
Score from 0.0 to 1.0:
  1.0 = Fully relevant, the answer directly and completely addresses the question
  0.5 = Partially relevant, the answer addresses some aspects but misses others
  0.0 = Not relevant, the answer does not address the question

Respond with ONLY a JSON object in this format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence justification>"}}"""

    CONTEXT_PRECISION_PROMPT = """\
You are an objective judge evaluating the relevance of a retrieved code chunk.

QUESTION: {question}

RETRIEVED CHUNK:
{chunk}

TASK: Is this code chunk relevant to answering the question?
Score: 1.0 if relevant, 0.0 if not relevant.

Respond with ONLY a JSON object: {{"score": <0.0 or 1.0>, "reason": "<brief reason>"}}"""

    def __init__(self, llm: OllamaClient, debug_log_dir: Optional[str] = None):
        self._llm = llm
        self._debug_log_dir = Path(debug_log_dir) if debug_log_dir else None
        if self._debug_log_dir:
            self._debug_log_dir.mkdir(parents=True, exist_ok=True)
        # Fallback diagnostics — how many score_* calls had to use the
        # neutral 0.5 because every parsing strategy failed twice in a row.
        self.total_evaluations = 0
        self.fallback_count = 0

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Report how often judge scoring fell back to the neutral 0.5."""
        rate = (self.fallback_count / self.total_evaluations) if self.total_evaluations else 0.0
        return {
            "total_evaluations": self.total_evaluations,
            "fallback_count": self.fallback_count,
            "fallback_rate": rate,
        }

    def score_faithfulness(
        self,
        question: str,
        answer: str,
        context_chunks: List[RetrievedChunk],
        query_id: Optional[str] = None,
    ) -> float:
        """Score answer faithfulness to the provided context [0, 1]."""
        context_text = "\n\n".join(
            f"[{i+1}] {c.chunk_type} `{c.function_name or c.class_name}` "
            f"from {c.file_path}:\n{c.raw_code[:500]}"
            for i, c in enumerate(context_chunks)
        )
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question,
            context=context_text[:4000],
            answer=answer[:2000],
        )
        return self._call_and_parse(prompt, metric="faithfulness", query_id=query_id)

    def score_relevancy(self, question: str, answer: str, query_id: Optional[str] = None) -> float:
        """Score answer relevancy to the question [0, 1]."""
        prompt = self.RELEVANCY_PROMPT.format(
            question=question,
            answer=answer[:2000],
        )
        return self._call_and_parse(prompt, metric="relevancy", query_id=query_id)

    def score_context_precision(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        query_id: Optional[str] = None,
    ) -> float:
        """
        Fraction of retrieved chunks that are relevant to the question.
        Scores each chunk individually and averages.
        """
        if not chunks:
            return 0.0

        scores = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.raw_code[:800]
            prompt = self.CONTEXT_PRECISION_PROMPT.format(
                question=question,
                chunk=chunk_text,
            )
            scores.append(
                self._call_and_parse(
                    prompt, metric=f"context_precision_chunk{i}", query_id=query_id
                )
            )

        return sum(scores) / len(scores)

    def score_context_recall(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        expected_answer: str,
        query_id: Optional[str] = None,
    ) -> float:
        """
        Estimate how well the retrieved context covers the expected answer.
        Uses a single LLM call to assess coverage.
        """
        context_text = "\n\n".join(
            f"[{i+1}] {c.chunk_type} from {c.file_path}:\n{c.raw_code[:400]}"
            for i, c in enumerate(chunks)
        )
        prompt = f"""You are evaluating context recall.

QUESTION: {question}

EXPECTED ANSWER: {expected_answer[:1000]}

RETRIEVED CONTEXT:
{context_text[:4000]}

TASK: What fraction of the information needed for the EXPECTED ANSWER is present in the RETRIEVED CONTEXT?
Score from 0.0 (none of the needed information is present) to 1.0 (all needed information is present).

Respond with ONLY a JSON object: {{"score": <float 0.0-1.0>, "reason": "<brief reason>"}}"""

        return self._call_and_parse(prompt, metric="context_recall", query_id=query_id)

    # ------------------------------------------------------------------
    # Generation + parsing (with retry) and debug logging
    # ------------------------------------------------------------------

    def _call_and_parse(
        self,
        prompt: str,
        metric: str,
        query_id: Optional[str] = None,
    ) -> float:
        """
        Generate a judge response and parse its score. Retries once (a
        fresh LLM call, not just a re-parse of the same text) if parsing
        fails, before falling back to a neutral 0.5. Every call — success
        or fallback — is counted toward fallback-rate diagnostics, and
        optionally written to a debug log for inspection.
        """
        self.total_evaluations += 1
        raw_text = ""
        score: Optional[float] = None
        parse_error: Optional[str] = None

        for attempt in range(1, 3):  # one real attempt + one retry
            try:
                response = self._llm.generate(prompt)
                raw_text = response.raw_text
            except Exception as exc:
                parse_error = f"LLM call failed (attempt {attempt}/2): {exc}"
                logger.warning("%s scoring (%s): %s", metric, query_id, parse_error)
                continue

            score = self._parse_score_strict(raw_text)
            if score is not None:
                break
            parse_error = f"Unparseable judge response (attempt {attempt}/2): {raw_text[:200]!r}"

        used_fallback = score is None
        if used_fallback:
            self.fallback_count += 1
            score = 0.5
            logger.warning(
                "%s: no parseable score for query_id=%s after 2 attempts — "
                "falling back to neutral 0.5. %s",
                metric, query_id, parse_error,
            )

        self._log_debug(
            query_id=query_id,
            metric=metric,
            prompt=prompt,
            raw_text=raw_text,
            score=score,
            error=parse_error if used_fallback else None,
        )
        return score

    def _log_debug(
        self,
        query_id: Optional[str],
        metric: str,
        prompt: str,
        raw_text: str,
        score: float,
        error: Optional[str],
    ) -> None:
        """Write prompt/raw-response/parsed-score to disk for debugging.
        No-op if debug_log_dir wasn't configured. Never raises — a
        logging failure must not break evaluation itself."""
        if not self._debug_log_dir:
            return
        try:
            safe_qid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(query_id or "unknown"))
            path = self._debug_log_dir / f"{safe_qid}_{metric}.txt"
            lines = [
                "Prompt", "-" * 16, prompt, "",
                "Raw Output", "-" * 16, raw_text, "",
                "Parsed", "-" * 16, str(score),
            ]
            if error:
                lines += ["", "Parse Error", "-" * 16, error]
            path.write_text("\n".join(lines), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write judge debug log for %s/%s: %s", query_id, metric, exc)

    # ------------------------------------------------------------------
    # Score parsing — multiple strategies, tried in order of strictness.
    # Returns None (not 0.5) when every strategy fails, so the caller can
    # decide to retry rather than silently accepting a guessed score.
    # ------------------------------------------------------------------

    def _parse_score_strict(self, text: str) -> Optional[float]:
        for candidate in (self._strip_code_fences(text), text):
            for strategy in (
                self._try_json_block,
                self._try_single_quote_json,
                self._try_key_value,
                self._try_bare_number,
            ):
                result = strategy(candidate)
                if result is not None:
                    return result
        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip ```json ... ``` or ``` ... ``` wrappers, if present."""
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _try_json_block(self, text: str) -> Optional[float]:
        """Strategy 1: find {...} block(s), parse as strict JSON."""
        for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
            try:
                data = json.loads(match.group())
                if "score" in data:
                    return self._clamp(float(data["score"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return None

    def _try_single_quote_json(self, text: str) -> Optional[float]:
        """Strategy 2: same, but tolerate single-quoted keys/values
        (e.g. {'score': 0.9}) — invalid JSON but a common model slip."""
        for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
            try:
                data = json.loads(match.group().replace("'", '"'))
                if "score" in data:
                    return self._clamp(float(data["score"]))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return None

    def _try_key_value(self, text: str) -> Optional[float]:
        """Strategy 3: bare 'score = 0.92' / 'Score: 0.92' outside any
        well-formed JSON object."""
        match = re.search(
            r'["\']?score["\']?\s*[:=]\s*["\']?([01](?:\.\d+)?)["\']?',
            text, re.IGNORECASE,
        )
        if match:
            try:
                return self._clamp(float(match.group(1)))
            except ValueError:
                pass
        return None

    def _try_bare_number(self, text: str) -> Optional[float]:
        """Strategy 4 (last resort before giving up): first bare
        float-like number in the response, e.g. a judge that just
        replies '0.8' with no surrounding structure."""
        match = re.search(r'\b(0\.\d+|1\.0|0|1)\b', text)
        if match:
            try:
                return self._clamp(float(match.group(1)))
            except ValueError:
                pass
        return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Comprehensive evaluation suite for PrivaRepo.

    Usage:
        evaluator = Evaluator(pipeline, config)
        report = evaluator.run_full_benchmark()
        evaluator.save_report(report, "eval_results/report.json")
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: Optional[EvalConfig] = None,
    ):
        self.pipeline = pipeline
        self.config = config or EvalConfig()
        self._judge = LLMJudge(
            pipeline.llm,
            debug_log_dir=str(Path(self.config.results_dir) / "judge_logs"),
        )
        Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Query loading
    # ------------------------------------------------------------------

    def load_eval_queries(self, path: Optional[str] = None) -> List[EvalQuery]:
        """Load evaluation queries from a JSON file."""
        query_path = Path(path or self.config.eval_queries_path)
        if not query_path.exists():
            logger.warning("Eval queries file not found: %s", query_path)
            return []

        with query_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        queries = []
        for item in data:
            queries.append(
                EvalQuery(
                    query_id=item.get("query_id", f"q_{len(queries)}"),
                    question=item["question"],
                    expected_answer=item.get("expected_answer"),
                    relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                    relevant_files=item.get("relevant_files", []),
                    relevant_functions=item.get("relevant_functions", []),
                    task_type=item.get("task_type", "general"),
                    language_filter=item.get("language_filter"),
                )
            )
        logger.info("Loaded %d evaluation queries from %s", len(queries), query_path)
        return queries

    # ------------------------------------------------------------------
    # Retrieval Benchmark
    # ------------------------------------------------------------------

    def run_retrieval_benchmark(
        self,
        queries: Optional[List[EvalQuery]] = None,
        k: Optional[int] = None,
    ) -> RetrievalBenchmarkResult:
        """
        Evaluate retrieval quality against ground-truth relevant chunk IDs.

        Computes Precision@K, Recall@K, MRR, and Hit Rate.
        """
        queries = queries or self.load_eval_queries()
        k = k or self.config.eval_k

        if not queries:
            raise ValueError("No evaluation queries available.")

        # Filter to queries that have ground truth chunk IDs or files
        graded_queries = [
            q for q in queries
            if q.relevant_chunk_ids or q.relevant_files or q.relevant_functions
        ]
        if not graded_queries:
            logger.warning(
                "No queries have ground truth data (relevant_chunk_ids/files/functions). "
                "Using file-based matching."
            )
            graded_queries = queries

        per_query_metrics: List[QueryRetrievalMetrics] = []

        for eq in graded_queries:
            logger.info("Evaluating retrieval for query: %s", eq.query_id)
            try:
                chunks, _ = self.pipeline.search(
                    query=eq.question,
                    language=eq.language_filter,
                )
                retrieved_ids = [c.chunk_id for c in chunks[:k]]
                retrieved_files = {Path(c.file_path).name for c in chunks[:k]}
                retrieved_fns = {c.function_name for c in chunks[:k] if c.function_name}

                # Build relevant set using all available ground truth (normalize files to basenames)
                relevant_ids = set(eq.relevant_chunk_ids)
                relevant_files = {Path(f).name for f in eq.relevant_files}
                relevant_fns = set(eq.relevant_functions)

                # Match chunks that hit on chunk ID, file basename, OR function name
                hit_chunk_ids = set()
                for c in chunks[:k]:
                    c_basename = Path(c.file_path).name
                    if (
                        c.chunk_id in relevant_ids
                        or c_basename in relevant_files
                        or (c.function_name and c.function_name in relevant_fns)
                    ):
                        hit_chunk_ids.add(c.chunk_id)

                total_matches = len(hit_chunk_ids)
                if relevant_ids:
                    n_relevant = len(relevant_ids)
                else:
                    n_relevant = max(1, len(relevant_files) + len(relevant_fns))

                precision = min(1.0, total_matches / k)
                recall = min(1.0, total_matches / n_relevant)
                hit = total_matches > 0

                # MRR: find rank of first relevant item
                rr = 0.0
                for rank, chunk in enumerate(chunks[:k], start=1):
                    c_basename = Path(chunk.file_path).name
                    is_relevant = (
                        chunk.chunk_id in relevant_ids
                        or c_basename in relevant_files
                        or (chunk.function_name and chunk.function_name in relevant_fns)
                    )
                    if is_relevant:
                        rr = 1.0 / rank
                        break

                per_query_metrics.append(
                    QueryRetrievalMetrics(
                        query_id=eq.query_id,
                        precision_at_k=precision,
                        recall_at_k=recall,
                        reciprocal_rank=rr,
                        hit=hit,
                        k=k,
                        retrieved_chunk_ids=retrieved_ids,
                        relevant_chunk_ids=eq.relevant_chunk_ids,
                    )
                )
            except Exception as exc:
                logger.error("Retrieval eval failed for %s: %s", eq.query_id, exc)

        if not per_query_metrics:
            raise RuntimeError("All retrieval evaluations failed.")

        precisions = [m.precision_at_k for m in per_query_metrics]
        recalls = [m.recall_at_k for m in per_query_metrics]
        rrs = [m.reciprocal_rank for m in per_query_metrics]
        hits = [m.hit for m in per_query_metrics]

        return RetrievalBenchmarkResult(
            precision_at_k=statistics.mean(precisions),
            recall_at_k=statistics.mean(recalls),
            mrr=statistics.mean(rrs),
            hit_rate=sum(hits) / len(hits),
            k=k,
            n_queries=len(per_query_metrics),
            precision_std=statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
            recall_std=statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            per_query=per_query_metrics,
        )

    # ------------------------------------------------------------------
    # Generation Benchmark
    # ------------------------------------------------------------------

    def run_generation_benchmark(
        self,
        queries: Optional[List[EvalQuery]] = None,
    ) -> GenerationBenchmarkResult:
        """
        Evaluate generation quality using LLM-as-judge for faithfulness and relevancy.
        """
        queries = queries or self.load_eval_queries()
        if not queries:
            raise ValueError("No evaluation queries available.")

        per_query_metrics: List[QueryGenerationMetrics] = []
        failed_query_ids: List[str] = []

        for eq in queries:
            logger.info("Evaluating generation for query: %s", eq.query_id)
            try:
                rag_response = self.pipeline.query(
                    question=eq.question,
                    task_type=eq.task_type,
                    language=eq.language_filter,
                )

                faithfulness = self._judge.score_faithfulness(
                    question=eq.question,
                    answer=rag_response.answer,
                    context_chunks=rag_response.retrieved_chunks,
                    query_id=eq.query_id,
                )

                relevancy = self._judge.score_relevancy(
                    question=eq.question,
                    answer=rag_response.answer,
                    query_id=eq.query_id,
                )

                ctx_precision = self._judge.score_context_precision(
                    question=eq.question,
                    chunks=rag_response.retrieved_chunks,
                    query_id=eq.query_id,
                )

                ctx_recall = 0.5  # Default if no expected answer
                if eq.expected_answer:
                    ctx_recall = self._judge.score_context_recall(
                        question=eq.question,
                        chunks=rag_response.retrieved_chunks,
                        expected_answer=eq.expected_answer,
                        query_id=eq.query_id,
                    )

                per_query_metrics.append(
                    QueryGenerationMetrics(
                        query_id=eq.query_id,
                        faithfulness=faithfulness,
                        answer_relevancy=relevancy,
                        context_precision=ctx_precision,
                        context_recall=ctx_recall,
                    )
                )

            except Exception as exc:
                logger.error("Generation eval failed for %s: %s", eq.query_id, exc)
                failed_query_ids.append(eq.query_id)

        if not per_query_metrics:
            raise RuntimeError("All generation evaluations failed.")

        fallback_stats = self._judge.get_fallback_stats()

        return GenerationBenchmarkResult(
            faithfulness=statistics.mean(m.faithfulness for m in per_query_metrics),
            answer_relevancy=statistics.mean(m.answer_relevancy for m in per_query_metrics),
            context_precision=statistics.mean(m.context_precision for m in per_query_metrics),
            context_recall=statistics.mean(m.context_recall for m in per_query_metrics),
            n_queries=len(per_query_metrics),
            faithfulness_std=(
                statistics.stdev(m.faithfulness for m in per_query_metrics)
                if len(per_query_metrics) > 1 else 0.0
            ),
            answer_relevancy_std=(
                statistics.stdev(m.answer_relevancy for m in per_query_metrics)
                if len(per_query_metrics) > 1 else 0.0
            ),
            per_query=per_query_metrics,
            n_failed=len(failed_query_ids),
            failed_query_ids=failed_query_ids,
            judge_total_evaluations=fallback_stats["total_evaluations"],
            judge_fallback_count=fallback_stats["fallback_count"],
            judge_fallback_rate=fallback_stats["fallback_rate"],
        )

    # ------------------------------------------------------------------
    # Latency Benchmark
    # ------------------------------------------------------------------

    def run_latency_benchmark(
        self,
        n_runs: Optional[int] = None,
        warmup_runs: int = 2,
    ) -> LatencyBenchmarkResult:
        """
        Measure retrieval, generation, and total latency over N benchmark runs.

        Uses a fixed set of benchmark queries (loaded from eval queries or defaults).
        Warmup runs are excluded from statistics.
        """
        n_runs = n_runs or self.config.benchmark_iterations
        queries = self.load_eval_queries()

        if not queries:
            # Use a set of hardcoded benchmark queries if no eval file exists
            queries = [
                EvalQuery(query_id="bench_0", question="How does authentication work?"),
                EvalQuery(query_id="bench_1", question="What functions handle database connections?"),
                EvalQuery(query_id="bench_2", question="Find the main entry point of the application"),
                EvalQuery(query_id="bench_3", question="What classes implement the interface pattern?"),
                EvalQuery(query_id="bench_4", question="How are errors handled in the HTTP client?"),
            ]

        retrieval_times: List[float] = []
        generation_times: List[float] = []
        total_times: List[float] = []
        n_failed = 0

        # Warmup now runs the full query() path (retrieval + generation),
        # not just search() — this warms Ollama's model load / keep_alive
        # too, so it doesn't contaminate the first real iteration.
        for i in range(warmup_runs):
            q = queries[i % len(queries)]
            try:
                self.pipeline.query(question=q.question)
            except Exception:
                pass

        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        for i in range(n_runs):
            q = queries[i % len(queries)]
            logger.info("Latency benchmark run %d/%d: %s", i + 1, n_runs, q.query_id)

            try:
                t_total = time.monotonic()
                rag_resp = self.pipeline.query(question=q.question)
                total_ms = (time.monotonic() - t_total) * 1000

                ret_time = getattr(rag_resp, "retrieval_time", None)
                gen_time = getattr(rag_resp, "generation_time", None)

                if ret_time is None or gen_time is None:
                    # QueryResponse doesn't expose per-stage timing under
                    # the expected names — fall back to a standalone
                    # search() call rather than silently recording a
                    # wrong or zeroed number. Logged once per occurrence
                    # so it's visible if this path is ever hit.
                    logger.warning(
                        "QueryResponse missing retrieval_time/generation_time "
                        "attributes; falling back to standalone search() timing "
                        "for this run (adds one redundant retrieval call)."
                    )
                    t_ret = time.monotonic()
                    self.pipeline.search(query=q.question)
                    ret_ms = (time.monotonic() - t_ret) * 1000
                    gen_ms = max(0.0, total_ms - ret_ms)
                else:
                    ret_ms = ret_time * 1000
                    gen_ms = gen_time * 1000

                retrieval_times.append(ret_ms)
                generation_times.append(gen_ms)
                total_times.append(total_ms)

            except Exception as exc:
                n_failed += 1
                logger.warning("Latency run %d failed: %s", i, exc)
                # Deliberately not appending anything here — a failed run
                # recorded as generation=0.0 would silently drag P50 down
                # and misrepresent real latency (this was the earlier bug).

        mem_after = process.memory_info().rss / 1024 / 1024
        memory_mb = mem_after - mem_before

        def percentile(data: List[float], p: int) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = math.ceil(p / 100 * len(sorted_data)) - 1
            return sorted_data[max(0, idx)]

        return LatencyBenchmarkResult(
            n_runs=len(retrieval_times),
            n_failed=n_failed,
            retrieval_p50_ms=percentile(retrieval_times, 50),
            retrieval_p95_ms=percentile(retrieval_times, 95),
            retrieval_p99_ms=percentile(retrieval_times, 99),
            generation_p50_ms=percentile(generation_times, 50),
            generation_p95_ms=percentile(generation_times, 95),
            generation_p99_ms=percentile(generation_times, 99),
            total_p50_ms=percentile(total_times, 50),
            total_p95_ms=percentile(total_times, 95),
            total_p99_ms=percentile(total_times, 99),
            memory_usage_mb=memory_mb,
            collection_size=self.pipeline.vector_store.count,
            raw_retrieval_ms=retrieval_times,
            raw_generation_ms=generation_times,
            raw_total_ms=total_times,
        )

    # ------------------------------------------------------------------
    # Full Benchmark
    # ------------------------------------------------------------------

    def run_full_benchmark(
        self,
        queries: Optional[List[EvalQuery]] = None,
        include_retrieval: bool = True,
        include_generation: bool = True,
        include_latency: bool = True,
    ) -> FullBenchmarkReport:
        """
        Run the complete evaluation suite and return a combined report.
        """
        queries = queries or self.load_eval_queries()

        system_info = {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }

        report = FullBenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            system_info=system_info,
            collection_stats=self.pipeline.get_stats(),
        )

        if include_retrieval and self.config.compute_retrieval_metrics:
            try:
                logger.info("Running retrieval benchmark...")
                report.retrieval = self.run_retrieval_benchmark(queries)
            except Exception as exc:
                logger.error("Retrieval benchmark failed: %s", exc)

        if include_generation and self.config.compute_generation_metrics:
            try:
                logger.info("Running generation benchmark...")
                report.generation = self.run_generation_benchmark(queries)
            except Exception as exc:
                logger.error("Generation benchmark failed: %s", exc)

        if include_latency and self.config.compute_latency_metrics:
            try:
                logger.info("Running latency benchmark...")
                report.latency = self.run_latency_benchmark()
            except Exception as exc:
                logger.error("Latency benchmark failed: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_report(
        self,
        report: FullBenchmarkReport,
        output_path: Optional[str] = None,
    ) -> str:
        """Serialise the benchmark report to JSON and return the file path."""
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                Path(self.config.results_dir) / f"benchmark_{ts}.json"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialise(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_serialise(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            return obj

        data = _serialise(report)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Benchmark report saved to %s", output_path)
        return str(output_path)

    def print_report_summary(self, report: FullBenchmarkReport) -> None:
        """Print a human-readable summary to the logger."""
        lines = [
            "=" * 60,
            "PRIVAREPO BENCHMARK REPORT",
            f"Timestamp: {report.timestamp}",
            f"Collection: {report.collection_stats.get('total_chunks', 0)} chunks",
            "=" * 60,
        ]

        if report.retrieval:
            r = report.retrieval
            lines += [
                "",
                "RETRIEVAL METRICS",
                f"  Precision@{r.k}:  {r.precision_at_k:.3f}",
                f"  Recall@{r.k}:     {r.recall_at_k:.3f}",
                f"  MRR:            {r.mrr:.3f}",
                f"  Hit Rate:       {r.hit_rate:.3f}",
            ]

        if report.generation:
            g = report.generation
            lines += [
                "",
                "GENERATION METRICS",
                f"  Faithfulness:     {g.faithfulness:.3f}",
                f"  Answer Relevancy: {g.answer_relevancy:.3f}",
                f"  Context Precision:{g.context_precision:.3f}",
                f"  Context Recall:   {g.context_recall:.3f}",
                f"  Failed Queries:   {g.n_failed}",
                f"  Judge Evaluations:{g.judge_total_evaluations}",
                f"  Judge Fallbacks:  {g.judge_fallback_count} ({g.judge_fallback_rate:.1%})",
            ]

        if report.latency:
            lt = report.latency
            lines += [
                "",
                "LATENCY (milliseconds)",
                f"  Retrieval P50/P95/P99: {lt.retrieval_p50_ms:.1f} / {lt.retrieval_p95_ms:.1f} / {lt.retrieval_p99_ms:.1f}",
                f"  Generation P50/P95:    {lt.generation_p50_ms:.1f} / {lt.generation_p95_ms:.1f}",
                f"  Total P50/P95:         {lt.total_p50_ms:.1f} / {lt.total_p95_ms:.1f}",
                f"  Memory Delta:          {lt.memory_usage_mb:.1f} MB",
                f"  Failed Runs:           {lt.n_failed}",
            ]

        lines.append("=" * 60)
        for line in lines:
            logger.info(line)