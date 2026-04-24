from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
# from envs.database_env.Lib import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

import requests
import yaml


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    endpoint: str | None = None
    timeout_seconds: int = 60


def load_config(config_path: str | Path | None = None) -> LLMConfig:
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"

    data: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                data = loaded

    cfg = data.get("llm", {}) if isinstance(data, dict) else {}
    if not isinstance(cfg, dict):
        cfg = {}

    return LLMConfig(
        provider=str(cfg.get("provider", LLMConfig.provider)),
        model=str(cfg.get("model", LLMConfig.model)),
        api_key_env=str(cfg.get("api_key_env", LLMConfig.api_key_env)),
        api_key=cfg.get("api_key") or None,
        endpoint=cfg.get("endpoint"),
        timeout_seconds=int(cfg.get("timeout_seconds", LLMConfig.timeout_seconds)),
    )


# =========================================================
# PROVIDER INTERFACE
# =========================================================

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        pass


# =========================================================
# OPENAI PROVIDER
# =========================================================

class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = config.endpoint or "https://api.openai.com/v1/chat/completions"

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError("Missing OpenAI API key")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.2,
        }

        response = requests.post(
            self.endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()


# =========================================================
# GOOGLE GEMINI PROVIDER
# =========================================================

class GeminiProvider(LLMProvider):
    """
    Uses Google Generative Language API (Gemini)
    Docs: https://ai.google.dev
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError("Missing Google Gemini API key")

        # Lazily import genai to avoid hard dependency until used
        try:
            from google import genai
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("The `genai` package is required for GeminiProvider. "
                               "Install it with `pip install genai`.") from exc

        model = self.config.model or "gemini-1.5-flash"

        # Gemini SDK doesn't use a separate system role; prepend if provided
        contents = f"SYSTEM: {system_prompt}\n\n{prompt}" if system_prompt else prompt

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=contents)

        # SDK response shapes vary by version; try common attributes
        text = getattr(response, "text", None)
        if text is not None:
            text = text.strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text

        # Try candidate-based shapes
        try:
            candidate_text = response.candidates[0].content[0].text
            if candidate_text:
                return candidate_text.strip()
        except Exception:
            pass

        try:
            candidate_text = response.candidates[0]["content"]["parts"][0]["text"]
            if candidate_text:
                return candidate_text.strip()
        except Exception:
            pass

        raise RuntimeError("Could not extract text from Gemini response")

# =========================================================
# MistralAI PROVIDER
# =========================================================

class MistralAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError("Missing MistralAI API key")

        # Initialize the MistralAI client
        client = ChatMistralAI(
            model=self.config.model,
            mistral_api_key=api_key,
            temperature=0.2
        )

        # Prepare the messages
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        # Get the response
        response = client.invoke(messages)
        return response.content

# =========================================================
# FACTORY
# =========================================================

def build_provider(config: LLMConfig) -> LLMProvider:
    provider = config.provider.lower()

    if provider == "openai":
        return OpenAIProvider(config)

    if provider in {"google", "gemini"}:
        return GeminiProvider(config)
    
    if provider in {"mistral", "mistralai"}:
        return MistralAIProvider(config)

    raise ValueError(f"Unsupported provider: {config.provider}")


# =========================================================
# CLIENT (CLEAN CORE INTERFACE)
# =========================================================

class LLMClient:
    def __init__(self, config_path: str | Path | None = None):
        config = load_config(config_path)
        # logging.info(f"[planning] LLMClient initialized with provider: {config.provider}, model: {config.model}")
        self.provider = build_provider(config)

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        return self.provider.complete(prompt, system_prompt)