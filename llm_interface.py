"""
llm_interface.py — Ollama LLM Client

Provides a robust, retry-aware interface to the Ollama REST API
with streaming support, response parsing, and connection health checking.

Design decisions:
  - Uses the `ollama` Python SDK for type safety and automatic streaming.
  - Retry with exponential backoff + jitter for transient connection errors.
  - Response is parsed from a structured format that the prompt enforces,
    extracting: answer, reasoning, referenced_files, functions_used.
  - Token estimation uses a simple heuristic (words × 1.3) since Ollama
    does not always expose token counts in non-streaming mode.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

import ollama
from ollama import Client, ResponseError

from config import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Structured output from the LLM, parsed from its formatted response."""

    raw_text: str
    answer: str
    reasoning: str = ""
    referenced_files: List[str] = field(default_factory=list)
    functions_used: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""

    @property
    def text(self) -> str:
        """Alias for raw_text, used by evaluator and pipeline callers."""
        return self.raw_text


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

class LLMResponseParser:
    """
    Parses the structured text output produced by PrivaRepo's system prompt.

    The LLM is instructed to respond in this format:

        ## ANSWER
        <answer text>

        ## REASONING
        <step-by-step reasoning>

        ## REFERENCED FILES
        - path/to/file.py
        - path/to/other.java

        ## FUNCTIONS USED
        - function_name (file.py)
        - other_function (module.py)

    We extract each section with regex and fall back gracefully if the LLM
    doesn't follow the format exactly.
    """

    _SECTION_RE = re.compile(
        r"##\s*(ANSWER|REASONING|REFERENCED FILES|FUNCTIONS USED)\s*\n(.*?)(?=##\s*(?:ANSWER|REASONING|REFERENCED FILES|FUNCTIONS USED)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    _LIST_ITEM_RE = re.compile(r"^\s*[-*•]\s*(.+)$", re.MULTILINE)

    def parse(self, raw_text: str) -> Dict[str, object]:
        """
        Parse LLM raw output into structured fields.
        Returns a dict with keys: answer, reasoning, referenced_files, functions_used.
        """
        sections: Dict[str, str] = {}
        for match in self._SECTION_RE.finditer(raw_text):
            key = match.group(1).strip().upper()
            value = match.group(2).strip()
            sections[key] = value

        answer = sections.get("ANSWER", "").strip()
        if not answer:
            # Fallback: use the entire text as the answer
            answer = raw_text.strip()

        reasoning = sections.get("REASONING", "").strip()

        referenced_files = self._extract_list(sections.get("REFERENCED FILES", ""))
        functions_used = self._extract_list(sections.get("FUNCTIONS USED", ""))

        return {
            "answer": answer,
            "reasoning": reasoning,
            "referenced_files": referenced_files,
            "functions_used": functions_used,
        }

    def _extract_list(self, text: str) -> List[str]:
        items = self._LIST_ITEM_RE.findall(text)
        return [item.strip() for item in items if item.strip()]


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Production-grade Ollama client with retry logic and streaming support.

    Usage:
        client = OllamaClient(config)
        response = client.generate(prompt)
        for token in client.stream_generate(prompt):
            print(token, end="", flush=True)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client = Client(host=self.config.base_url)
        self._parser = LLMResponseParser()
        logger.info(
            "OllamaClient initialised: model='%s', base_url='%s'",
            self.config.model,
            self.config.base_url,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            # Check if the configured model is available (handle tag variants)
            model_base = self.config.model.split(":")[0]
            for m in available:
                if m.startswith(model_base):
                    return True
            logger.warning(
                "Model '%s' not found in Ollama. Available: %s",
                self.config.model,
                available,
            )
            return False
        except Exception as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def list_models(self) -> List[str]:
        """Return list of models available in Ollama."""
        try:
            models = self._client.list()
            return [m.model for m in models.models]
        except Exception as exc:
            logger.error("Failed to list Ollama models: %s", exc)
            return []

    def pull_model(self, model: Optional[str] = None) -> None:
        """Pull a model from Ollama registry if not locally available."""
        model = model or self.config.model
        logger.info("Pulling model '%s' from Ollama...", model)
        self._client.pull(model)
        logger.info("Model '%s' pulled successfully.", model)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a response from Ollama with retry on transient failures.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system/context prompt.

        Returns:
            Structured LLMResponse.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    keep_alive=self.config.keep_alive,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "num_ctx": self.config.context_window,
                    },
                )
                latency = time.monotonic() - t0

                raw_text = response.message.content or ""
                parsed = self._parser.parse(raw_text)

                # Token estimates (Ollama sometimes returns these)
                prompt_tokens = getattr(response, "prompt_eval_count", None) or self._estimate_tokens(prompt)
                completion_tokens = getattr(response, "eval_count", None) or self._estimate_tokens(raw_text)

                return LLMResponse(
                    raw_text=raw_text,
                    answer=parsed["answer"],
                    reasoning=parsed["reasoning"],
                    referenced_files=parsed["referenced_files"],
                    functions_used=parsed["functions_used"],
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=latency,
                    model=self.config.model,
                )

            except (ResponseError, ConnectionError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, self.config.max_retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Ollama request failed after %d attempts: %s",
                        self.config.max_retries, exc,
                    )

        raise RuntimeError(
            f"Ollama generation failed after {self.config.max_retries} retries: {last_exc}"
        )

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream tokens from Ollama as they are generated.

        Yields individual token strings for real-time output.
        Use in the interactive CLI for responsive UX.

        Usage:
            for token in client.stream_generate(prompt):
                print(token, end="", flush=True)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self._client.chat(
                model=self.config.model,
                messages=messages,
                stream=True,
                keep_alive=self.config.keep_alive,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "num_ctx": self.config.context_window,
                },
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"\n\n[ERROR: Streaming failed — {exc}]"

    def generate_with_history(
        self,
        messages: List[Dict[str, str]],
    ) -> LLMResponse:
        """
        Generate with a full conversation history for interactive chat mode.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}.

        Returns:
            Structured LLMResponse.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                t0 = time.monotonic()
                response = self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    keep_alive=self.config.keep_alive,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "num_ctx": self.config.context_window,
                    },
                )
                latency = time.monotonic() - t0
                raw_text = response.message.content or ""
                parsed = self._parser.parse(raw_text)
                return LLMResponse(
                    raw_text=raw_text,
                    answer=parsed["answer"],
                    reasoning=parsed["reasoning"],
                    referenced_files=parsed["referenced_files"],
                    functions_used=parsed["functions_used"],
                    latency_seconds=latency,
                    model=self.config.model,
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self._backoff_delay(attempt))

        raise RuntimeError(f"Chat failed: {last_exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^attempt ± 10% jitter."""
        base = self.config.retry_base_delay * (2 ** (attempt - 1))
        jitter = base * random.uniform(-0.1, 0.1)
        return min(base + jitter, 30.0)  # Cap at 30s

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate: ~1.3 tokens per word."""
        return int(len(text.split()) * 1.3)