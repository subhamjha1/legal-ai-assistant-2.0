"""
LLM providers for answer generation (Milestone 7).

Why one interface, three providers:
The assignment spec lists GPT-4.1, Claude, and Gemini as acceptable LLMs.
Rather than hard-code one, `LLMProvider` lets the QA service call any of
them identically - only `generate()` matters to callers.

What's actually verified in this sandbox, and why that differs from
Milestones 3/4/6's gaps:
- `api.anthropic.com` IS reachable from this sandbox's network allow-list
  (confirmed: a request returns 404, not a proxy-blocked 403). So
  `AnthropicProvider` is real, working code you could call directly - the
  only missing piece is a real `ANTHROPIC_API_KEY`, which this sandbox
  doesn't have configured. That's a credentials gap, not a network gap.
- `OpenAIProvider` and `GeminiProvider` are complete production code but
  their APIs (`api.openai.com`, `generativelanguage.googleapis.com`) are
  outside this sandbox's network allow-list entirely (confirmed 403 for
  OpenAI in Milestone 3) - the same kind of gap as BGE/cross-encoder model
  downloads.

Testing strategy: `FakeLLMProvider` (in tests/) lets the QA service's
prompt construction, citation formatting, and no-evidence handling be
fully verified without needing any real API key or network call at all -
that logic is what actually prevents hallucination, and it's provider-
agnostic by design.
"""
from abc import ABC, abstractmethod
from typing import Iterator

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.qa import TokenUsage

logger = get_logger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_message: str) -> str:
        """Return the model's raw text response."""

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Yield the model's response incrementally, as text deltas."""

    def generate_with_usage(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage | None]:
        """
        Like generate(), but also returns token usage when the provider can
        report it, without making a second API call. Used by QAService
        (for cost visibility) and the evaluation harness (Milestone 9,
        token usage metric).

        Default implementation falls back to generate() with usage=None -
        safe for any provider (including test fakes) that doesn't override
        this; real providers below override it to pull usage directly off
        the same API response they already made.
        """
        return self.generate(system_prompt, user_message), None


class AnthropicProvider(LLMProvider):
    """Real, network-reachable from this sandbox - only needs a valid
    ANTHROPIC_API_KEY to run end-to-end."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required to use the 'anthropic' LLM provider. "
                "Set it in .env, or switch LLM_PROVIDER to 'openai' or 'gemini'."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        text, _ = self.generate_with_usage(system_prompt, user_message)
        return text

    def generate_with_usage(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage | None]:
        response = self._client.messages.create(
            model=self.settings.anthropic_qa_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
        )
        return text, usage

    def generate_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.settings.anthropic_qa_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            yield from stream.text_stream


class OpenAIProvider(LLMProvider):
    """Real production code; api.openai.com is outside this sandbox's
    network allow-list (see module docstring). Verify on a machine with
    normal internet access and a real OPENAI_API_KEY."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to use the 'openai' LLM provider. "
                "Set it in .env, or switch LLM_PROVIDER to 'anthropic' or 'gemini'."
            )
        from openai import OpenAI
        self._client = OpenAI(api_key=self.settings.openai_api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        text, _ = self.generate_with_usage(system_prompt, user_message)
        return text

    def generate_with_usage(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage | None]:
        response = self._client.chat.completions.create(
            model=self.settings.openai_qa_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens
            )
        return text, usage

    def generate_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.settings.openai_qa_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiProvider(LLMProvider):
    """Real production code; generativelanguage.googleapis.com is outside
    this sandbox's network allow-list. Verify on a machine with normal
    internet access and a real GEMINI_API_KEY.

    Uses the current `google-genai` SDK (the older `google-generativeai`
    package is deprecated as of this writing)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to use the 'gemini' LLM provider. "
                "Set it in .env, or switch LLM_PROVIDER to 'anthropic' or 'openai'."
            )
        from google import genai
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def generate(self, system_prompt: str, user_message: str) -> str:
        text, _ = self.generate_with_usage(system_prompt, user_message)
        return text

    def generate_with_usage(self, system_prompt: str, user_message: str) -> tuple[str, TokenUsage | None]:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.settings.gemini_qa_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            ),
        )
        usage = None
        if response.usage_metadata:
            usage = TokenUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )
        return response.text, usage

    def generate_stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        from google.genai import types

        stream = self._client.models.generate_content_stream(
            model=self.settings.gemini_qa_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.settings.llm_temperature,
                max_output_tokens=self.settings.llm_max_tokens,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    settings = get_settings()
    name = provider_name or settings.llm_provider

    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"Unknown LLM provider '{name}'. Expected 'anthropic', 'openai', or 'gemini'.")
