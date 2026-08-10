from __future__ import annotations

import logging
import time
from functools import lru_cache

from openai import OpenAI

from backend.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self._client = OpenAI(
            base_url=settings.llm_api_base,
            api_key=settings.resolved_llm_api_key or "not-set",
            timeout=settings.llm_request_timeout,
        )
        self.model = settings.llm_model_name

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        retries: int = 2,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                start = time.monotonic()
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or settings.llm_max_tokens,
                    temperature=temperature if temperature is not None else settings.llm_temperature,
                )
                elapsed = time.monotonic() - start
                content = resp.choices[0].message.content or ""
                logger.info(
                    "LLM call ok model=%s tokens_in~%d elapsed=%.2fs", self.model, len(prompt) // 4, elapsed
                )
                return content
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM call failed after {retries + 1} attempts") from last_error

    def stream(self, prompt: str, system: str | None = None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()
