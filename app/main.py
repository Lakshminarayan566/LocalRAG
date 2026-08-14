"""
main.py — PrivaRepo FastAPI application entrypoint.

Directory layout this expects (backend/ sits INSIDE the existing
PrivaRepo project, alongside the files it wraps):

    privarepo/
    ├── config.py
    ├── rag_pipeline.py
    ├── vector_store.py
    ├── llm_interface.py
    ├── tree_sitter_chunker.py
    ├── repository_manager.py
    ├── repositories.json          (created at runtime by RepositoryManager)
    └── backend/                   <- this FastAPI app
        ├── main.py
        ├── dependencies.py
        ├── core/
        ├── schemas/
        └── routers/

Run from the privarepo/ project root:
    uvicorn backend.main:app --reload --port 8000

The sys.path insertion below makes `from config import AppConfig` (and
similarly for rag_pipeline / vector_store / llm_interface /
repository_manager) resolve correctly regardless of the working
directory uvicorn is launched from.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging  # noqa: E402  (must follow the sys.path fix above)

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from fastapi import HTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from .core.logging_config import setup_logging
from .routers import chat, health, indexing, repositories, search, settings
from .schemas.common import ErrorResponse

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PrivaRepo API",
    description="REST/SSE API wrapping the existing PrivaRepo RAG backend "
                 "(rag_pipeline.py, vector_store.py, llm_interface.py, "
                 "repository_manager.py). No retrieval, indexing, or "
                 "generation logic lives in this API layer — see each "
                 "router module's docstring for exactly which existing "
                 "backend function each endpoint calls.",
    version="1.0.0",
)

# Permissive by default for local development (this mirrors config.py's
# own APIConfig.cors_origins default of ["*"]) — tighten before any
# non-local deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Every raise HTTPException(...) in the routers goes through here so
    the response body is always {error, message, detail} instead of
    FastAPI's bare default {"detail": "..."}."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"http_{exc.status_code}",
            message=str(exc.detail),
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Pydantic request-body validation failures (e.g. missing required
    field, wrong type) -> the same ErrorResponse shape as everything else."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed.",
            detail=str(exc.errors()),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected backend exception never returns a bare
    500 with no body — every error response uses the same ErrorResponse
    shape, whether raised explicitly (HTTPException in a router) or not."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An unexpected error occurred.",
            detail=str(exc),
        ).model_dump(),
    )


app.include_router(health.router)
app.include_router(repositories.router)
app.include_router(indexing.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(settings.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"name": "PrivaRepo API", "docs": "/docs"}
