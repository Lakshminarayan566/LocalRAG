from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Whitelist of AppConfig fields this API allows patching, mapped as
# "section.field" -> the dotted path PipelineManager.apply_settings()
# expects. Deliberately excludes structural fields (persist_dir,
# collection_name, bm25_index_dir) — those are owned by repository
# scoping in RAGPipeline.__init__ / repository_manager.py, not by
# runtime settings, and mutating them here would fight that logic.
MUTABLE_FIELDS = {
    "llm_model": "llm.model",
    "llm_temperature": "llm.temperature",
    "llm_max_tokens": "llm.max_tokens",
    "llm_context_window": "llm.context_window",
    "llm_keep_alive": "llm.keep_alive",
    "top_k_vector": "retrieval.top_k_vector",
    "top_k_bm25": "retrieval.top_k_bm25",
    "rerank_candidates": "retrieval.rerank_candidates",
    "final_top_k": "retrieval.final_top_k",
    "use_reranker": "retrieval.use_reranker",
}


class SettingsUpdateRequest(BaseModel):
    """Every field is optional — only the ones provided get patched."""

    llm_model: Optional[str] = None
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    llm_max_tokens: Optional[int] = Field(None, gt=0)
    llm_context_window: Optional[int] = Field(None, gt=0)
    llm_keep_alive: Optional[str] = None
    top_k_vector: Optional[int] = Field(None, gt=0)
    top_k_bm25: Optional[int] = Field(None, gt=0)
    rerank_candidates: Optional[int] = Field(None, gt=0)
    final_top_k: Optional[int] = Field(None, gt=0)
    use_reranker: Optional[bool] = None

    def to_dotted_updates(self) -> dict:
        """Convert only the fields the caller actually set into the
        {"section.field": value} shape PipelineManager.apply_settings()
        expects."""
        result = {}
        for api_name, dotted_path in MUTABLE_FIELDS.items():
            value = getattr(self, api_name)
            if value is not None:
                result[dotted_path] = value
        return result


class SettingsResponse(BaseModel):
    """Mirrors AppConfig.summary()'s existing fields, plus the few extra
    ones (max_tokens, keep_alive) that matter for the API's settings UI
    but aren't in that method's return dict."""

    embedding_model: str
    collection: str
    persist_dir: str
    llm_model: str
    llm_max_tokens: int
    llm_temperature: float
    llm_keep_alive: str
    top_k_vector: int
    top_k_bm25: int
    rrf_k: int
    rerank_candidates: int
    final_top_k: int
    use_reranker: bool
