import asyncio
import json
import math
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.pipeline_manager import PipelineManager
from ..dependencies import get_pipeline_manager
# PromptBuilder / LLMResponseParser are reused exactly as the existing
# blocking path uses them (via OllamaClient.generate() internally) — no
# new prompt content, no new parsing logic. Adjust this import path if
# rag_pipeline.py doesn't sit directly under app/ in your project.
from rag_pipeline import PromptBuilder
from llm_interface import LLMResponseParser

def _run_sync_generator(sync_gen, loop: asyncio.AbstractEventLoop, q: "asyncio.Queue", sentinel: object) -> None:
    """Runs on a background thread. Pushes each item from a blocking sync
    generator onto an asyncio.Queue via call_soon_threadsafe."""
    try:
        for item in sync_gen:
            loop.call_soon_threadsafe(q.put_nowait, item)
    except Exception as exc:  # surfaced to the async consumer, not swallowed
        loop.call_soon_threadsafe(q.put_nowait, exc)
    finally:
        loop.call_soon_threadsafe(q.put_nowait, sentinel)


async def _iter_sync_generator_in_thread(sync_gen) -> AsyncGenerator[Any, None]:
    """
    Bridges a blocking synchronous generator onto an async generator.

    OllamaClient.stream_generate() is a plain `def` using the `ollama`
    SDK's blocking HTTP streaming — consuming it directly inside an
    `async def` would block FastAPI's event loop for the entire
    generation (the ~50s+ you're seeing today). This runs it on a
    background thread and relays items back via a queue instead. Fresh
    queue/thread per call, so concurrent requests never share state.
    """
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    threading.Thread(
        target=_run_sync_generator, args=(sync_gen, loop, q, _SENTINEL), daemon=True
    ).start()

    while True:
        item = await q.get()
        if item is _SENTINEL:
            return
        if isinstance(item, Exception):
            raise item
        yield item


# ---------------------------------------------------------------------------
# Streaming header classification
#
# PrivaRepo's structured output format uses whole-line section headers:
#   ## ANSWER
#   ## REASONING
#   ## REFERENCED FILES
#   ## FUNCTIONS USED
#
# The previous implementation re-ran LLMResponseParser.parse() (a
# whole-response parser) against the growing `raw_text` after every
# single token. That parser treats "current end of string" as if it
# were "end of full response", which is only true once, at the very
# end of the stream. Mid-stream it isn't true, and that mismatch is
# exactly what let raw "## ANSWER" text and partial next-header
# fragments (e.g. "## REASON") leak into the visible answer:
#   - LLMResponseParser._SECTION_RE requires a newline after the header
#     keyword before it recognizes a section, but the old
#     _ANSWER_HEADER_RE flipped `answer_started = True` as soon as the
#     bare substring "## ANSWER" appeared — before that newline arrived.
#     In that window `_SECTION_RE` matched nothing, so parse()'s
#     no-match fallback (`answer = raw_text.strip()`) fired and handed
#     back the literal header text as if it were answer content.
#   - _SECTION_RE's non-greedy `(.*?)` for the ANSWER section only stops
#     at a *complete* next-header keyword or `$`. While "## REASONING"
#     was still arriving token-by-token, raw_text could end in
#     "## REASON" — not a complete keyword, so the `$` branch matched
#     right there and the partial header text got folded into "answer".
#
# The fix below never calls LLMResponseParser mid-stream. It classifies
# only complete lines (a header can only be recognized once it's fully
# present), so there is no window where a partial header can be
# misread as content. LLMResponseParser is still used exactly once, on
# the final complete raw_text, for referenced_files/functions_used —
# unchanged from before.
# ---------------------------------------------------------------------------

_HEADER_TEXTS = ("## ANSWER", "## REASONING", "## REFERENCED FILES", "## FUNCTIONS USED")
_HEADER_LINE_RE = re.compile(
    r"^\s*##\s*(ANSWER|REASONING|REFERENCED FILES|FUNCTIONS USED)\s*$",
    re.IGNORECASE,
)


def _looks_like_header_prefix(partial_line: str) -> bool:
    """True if `partial_line` (a line with no closing '\\n' yet) could
    still turn into one of the structured section headers as more
    tokens arrive. Used to decide whether it's safe to stream it to the
    client yet, or whether it must stay buffered a little longer."""
    stripped = partial_line.lstrip()
    if not stripped:
        return True
    upper = stripped.upper()
    return any(h.startswith(upper) or upper.startswith(h) for h in _HEADER_TEXTS)


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query or context-aware prompt")
    repo_name: Optional[str] = Field(
        None, description="Repository name; defaults to active repository"
    )
    task_focus: Optional[str] = Field(
        "general",
        description=(
            "Task type: general, explain, find_bugs, similar_code, "
            "function_search, class_search"
        ),
    )

    # Kept for frontend compatibility.
    # Current RAGPipeline.query() does not accept these directly;
    # retrieval/model settings remain controlled by AppConfig.
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(5, ge=1, le=20)
    use_reranker: Optional[bool] = True

    language: Optional[str] = None
    chunk_type: Optional[str] = None
    file_path: Optional[str] = None
    class_name: Optional[str] = None


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    code_content: str
    score: float
    language: str
    function_name: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunkOut]
    referenced_files: List[str]
    functions_used: List[str]
    timings_ms: Dict[str, float]


def _chunk_to_output(chunk) -> RetrievedChunkOut:
    """Convert the real rag_pipeline.RetrievedChunk to the frontend schema."""

    # Prefer the strongest available ranking score.
    score = getattr(chunk, "rerank_score", 0.0)
    if not score:
        score = getattr(chunk, "rrf_score", 0.0)
    if not score:
        score = getattr(chunk, "vector_score", 0.0)

    score = float(score or 0.0)
    # Guard against non-finite scores (e.g. NaN from the cross-encoder
    # reranker on some repositories). `not score` above treats NaN as
    # truthy (NaN is never falsy in Python), so a NaN rerank_score would
    # otherwise survive all the way to json.dumps(), which serialises it
    # as the literal token `NaN` — not valid JSON. The frontend's
    # JSON.parse() throws on that token, which silently drops the whole
    # "chunks" SSE event (the separate "token" event has no score field,
    # so the answer still renders) — this is what makes retrieved context
    # disappear from the UI even though generation succeeded. Falling
    # back to 0.0 keeps the SSE payload parseable no matter what the
    # reranker returns.
    if math.isnan(score) or math.isinf(score):
        score = 0.0

    return RetrievedChunkOut(
        chunk_id=str(getattr(chunk, "chunk_id", "")),
        file_path=str(getattr(chunk, "file_path", "")),
        start_line=int(getattr(chunk, "start_line", 0)),
        end_line=int(getattr(chunk, "end_line", 0)),
        code_content=str(getattr(chunk, "raw_code", "")),
        score=score,
        language=str(getattr(chunk, "language", "")),
        function_name=getattr(chunk, "function_name", None),
    )


def _rag_to_response(rag_result) -> ChatResponse:
    chunks = [
        _chunk_to_output(chunk)
        for chunk in (rag_result.retrieved_chunks or [])
    ]

    return ChatResponse(
        answer=rag_result.answer,
        retrieved_chunks=chunks,
        referenced_files=list(rag_result.referenced_files or []),
        functions_used=list(rag_result.functions_used or []),
        timings_ms={
            "retrieval_ms": round(rag_result.retrieval_time * 1000, 2),
            "generation_ms": round(rag_result.generation_time * 1000, 2),
            "total_ms": round(rag_result.total_time * 1000, 2),
        },
    )


@router.post("/sync", response_model=ChatResponse)
async def chat_sync(
    payload: ChatRequest,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """Run the real PrivaRepo RAG pipeline and return the full answer."""

    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    try:
        pipeline = await manager.get_pipeline(payload.repo_name)

        # RAGPipeline.query() accepts these exact arguments.
        rag_result = await asyncio.to_thread(
            pipeline.query,
            question=payload.query,
            task_type=payload.task_focus or "general",
            language=payload.language,
            chunk_type=payload.chunk_type,
            file_path=payload.file_path,
            class_name=payload.class_name,
        )

        return _rag_to_response(rag_result)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


@router.post("")
async def chat_stream(
    payload: ChatRequest,
    manager: PipelineManager = Depends(get_pipeline_manager),
):
    """
    True token-by-token SSE streaming.

    Event flow:
      1. pipeline.search()            -> retrieval (unchanged retrieval code)
      2. PromptBuilder().build_prompt() -> same prompt content query() uses
      3. pipeline.llm.stream_generate() -> real Ollama token stream, bridged
         off the event loop via _iter_sync_generator_in_thread()
      4. Tokens are accumulated and split into complete lines. Each
         complete line is checked against the structured section
         headers ("## ANSWER" / "## REASONING" / "## REFERENCED FILES"
         / "## FUNCTIONS USED"). Only lines that fall inside the
         ANSWER section are streamed to the client; header lines
         themselves are never emitted, and streaming stops the instant
         a non-ANSWER header line is recognized. REASONING /
         REFERENCED FILES / FUNCTIONS USED are still extracted the same
         way as before, from the final accumulated text, via the
         existing LLMResponseParser, for the 'done' event.

    NOTE (see chat message accompanying this patch): pipeline.search()
    does not accept file_path/class_name the way pipeline.query() does,
    and does not run the architecture-question diversification that
    lives inside query(). This is a direct consequence of not modifying
    rag_pipeline.py this round — flagged, not hidden.
    """

    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        total_start = time.perf_counter()

        try:
            pipeline = await manager.get_pipeline(payload.repo_name)

            # --- Retrieval (unchanged retrieval code, just called
            # directly instead of via query()'s blocking bundle) ---
            t_ret_start = time.perf_counter()
            chunks, _retrieval_internal_timings = await asyncio.to_thread(
                pipeline.search,
                query=payload.query,
                language=payload.language,
                chunk_type=payload.chunk_type,
            )
            retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

            chunk_dicts = [_chunk_to_output(c).model_dump() for c in chunks]
            yield (
                "event: chunks\n"
                f"data: {json.dumps({'results': chunk_dicts, 'timings_ms': {'retrieval': round(retrieval_ms, 2)}})}\n\n"
            )

            # --- Prompt construction (same content query() builds) ---
            prompt = PromptBuilder().build_prompt(
                question=payload.query,
                retrieved_chunks=chunks,
                task_type=payload.task_focus or "general",
            )

            # --- Real token streaming, filtered to the ANSWER section ---
            # Fresh, request-local state — no shared mutable state
            # between concurrent requests.
            parser = LLMResponseParser()
            raw_text = ""
            line_buffer = ""   # unflushed text of the current, not-yet-'\n'-terminated line
            mode = "PRE"       # PRE -> waiting for "## ANSWER"; ANSWER -> streaming; DONE -> stop

            t_gen_start = time.perf_counter()
            sync_stream = pipeline.llm.stream_generate(
                prompt=prompt, system_prompt=PromptBuilder.SYSTEM_PROMPT,
            )
            async for token in _iter_sync_generator_in_thread(sync_stream):
                raw_text += token
                if mode == "DONE":
                    continue

                line_buffer += token

                # Drain every complete line currently in the buffer.
                while True:
                    newline_idx = line_buffer.find("\n")
                    if newline_idx == -1:
                        break
                    line = line_buffer[:newline_idx]
                    line_buffer = line_buffer[newline_idx + 1:]

                    header_match = _HEADER_LINE_RE.match(line)
                    if header_match:
                        section = header_match.group(1).upper()
                        if section == "ANSWER":
                            mode = "ANSWER"
                            continue
                        # REASONING / REFERENCED FILES / FUNCTIONS USED:
                        # stop streaming immediately, never emit this line.
                        mode = "DONE"
                        line_buffer = ""
                        break

                    if mode == "ANSWER":
                        yield (
                            "event: token\n"
                            f"data: {json.dumps({'text': line + chr(10)})}\n\n"
                        )
                    # mode == "PRE" and not a header line: preamble text
                    # before "## ANSWER" appears yet — intentionally
                    # dropped, matching the original buffering-preamble
                    # behaviour.

                if mode == "DONE":
                    continue

                # Stream the still-open partial line as soon as it can no
                # longer turn into a header, so output still feels
                # token-by-token outside of header boundaries.
                if mode == "ANSWER" and line_buffer and not _looks_like_header_prefix(line_buffer):
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'text': line_buffer})}\n\n"
                    )
                    line_buffer = ""

            generation_ms = (time.perf_counter() - t_gen_start) * 1000

            # Final parse over the now-complete raw_text is unambiguous
            # (no more tokens are coming), so LLMResponseParser is safe
            # to use here exactly as it always was — referenced_files /
            # functions_used / reasoning are unaffected by this patch.
            final_parsed = parser.parse(raw_text)

            if mode == "PRE":
                # Model never produced a recognizable "## ANSWER" header
                # line at all — preserve the parser's existing
                # whole-text fallback so behaviour when the model
                # doesn't follow the format is unchanged from before
                # this patch (show something rather than nothing).
                fallback_answer = final_parsed["answer"]
                if fallback_answer:
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'text': fallback_answer})}\n\n"
                    )
            elif mode == "ANSWER" and line_buffer:
                # Trailing partial line with no closing newline: the
                # stream has ended, so it can no longer become a
                # header — flush it now.
                yield (
                    "event: token\n"
                    f"data: {json.dumps({'text': line_buffer})}\n\n"
                )

            total_ms = (time.perf_counter() - total_start) * 1000

            yield (
                "event: done\n"
                f"data: {json.dumps({'retrieval_time': round(retrieval_ms, 2), 'generation_time': round(generation_ms, 2), 'total_time': round(total_ms, 2), 'referenced_files': final_parsed['referenced_files'], 'functions_used': final_parsed['functions_used']})}\n\n"
            )

        except Exception as exc:
            yield (
                "event: error\n"
                f"data: {json.dumps({'message': str(exc)})}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )