from __future__ import annotations


class LLMClient:
    """Thin wrapper around the language model provider used across the pipeline."""

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise NotImplementedError("LLM integration is not implemented yet.")
