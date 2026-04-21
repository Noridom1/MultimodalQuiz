from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    timeout_seconds: int = 60


class LLMClient:
    """Thin wrapper around the language model provider used across the pipeline."""

    def __init__(self, *, config_path: str | Path | None = None) -> None:
        self._config = self._load_config(config_path)

    @staticmethod
    def _default_config_path() -> Path:
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "configs" / "model_config.yaml"

    def _load_config(self, config_path: str | Path | None) -> LLMConfig:
        selected_path = Path(config_path) if config_path else self._default_config_path()
        payload: dict[str, Any] = {}

        if selected_path.exists():
            with selected_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                payload = loaded

        llm_cfg = payload.get("llm", {}) if isinstance(payload, dict) else {}
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}

        return LLMConfig(
            provider=str(llm_cfg.get("provider", LLMConfig.provider)),
            model=str(llm_cfg.get("model", LLMConfig.model)),
            api_key_env=str(llm_cfg.get("api_key_env", LLMConfig.api_key_env)),
            api_key=(
                str(llm_cfg.get("api_key"))
                if llm_cfg.get("api_key") not in (None, "")
                else LLMConfig.api_key
            ),
            endpoint=str(llm_cfg.get("endpoint", LLMConfig.endpoint)),
            timeout_seconds=int(llm_cfg.get("timeout_seconds", LLMConfig.timeout_seconds)),
        )

    def _resolve_api_key(self) -> str | None:
        if self._config.api_key:
            return self._config.api_key
        return os.environ.get(self._config.api_key_env)

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if self._config.provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {self._config.provider}")

        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set '{self._config.api_key_env}' or provide llm.api_key in config."
            )

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Prompt cannot be empty.")

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": cleaned_prompt})

        body = {
            "model": self._config.model,
            "messages": messages,
            "temperature": 0.2,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self._config.endpoint,
                json=body,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError("LLM request failed.") from exc
        except ValueError as exc:
            raise RuntimeError("LLM returned non-JSON response payload.") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("LLM response missing choices.")

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            joined = "\n".join(part.strip() for part in parts if part.strip()).strip()
            if joined:
                return joined

        raise RuntimeError("LLM response missing assistant text content.")
