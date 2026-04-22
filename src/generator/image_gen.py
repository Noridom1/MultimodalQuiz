from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import requests
import yaml


@dataclass(frozen=True)
class ImageGenerationConfig:
    provider: str = "imagerouter"
    endpoint: str = "https://api.imagerouter.io/v1/openai/images/edits"
    model: str = "google/nano-banana-2:free"
    quality: str = "auto"
    size: str = "auto"
    response_format: str = "url"
    output_format: str = "jpeg"
    api_key_env: str = "IMAGEROUTER_API_KEY"
    api_key: str | None = None
    timeout_seconds: int = 120
    default_images: tuple[str, ...] = ()
    default_mask: str | None = None


class ImageGenerator:
    def __init__(self, pipeline=None, *, config_path: str | Path | None = None):
        self.pipeline = pipeline
        self._config = self._load_config(config_path)

    @staticmethod
    def _default_config_path() -> Path:
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "configs" / "model_config.yaml"

    def _load_config(self, config_path: str | Path | None) -> ImageGenerationConfig:
        selected_path = Path(config_path) if config_path else self._default_config_path()
        payload: dict[str, Any] = {}

        if selected_path.exists():
            with selected_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                payload = loaded

        image_cfg = payload.get("image_generation", {}) if isinstance(payload, dict) else {}
        if not isinstance(image_cfg, dict):
            image_cfg = {}

        default_images = image_cfg.get("default_images", [])
        if not isinstance(default_images, list):
            default_images = []

        return ImageGenerationConfig(
            provider=str(image_cfg.get("provider", ImageGenerationConfig.provider)),
            endpoint=str(image_cfg.get("endpoint", ImageGenerationConfig.endpoint)),
            model=str(image_cfg.get("model", ImageGenerationConfig.model)),
            quality=str(image_cfg.get("quality", ImageGenerationConfig.quality)),
            size=str(image_cfg.get("size", ImageGenerationConfig.size)),
            response_format=str(image_cfg.get("response_format", ImageGenerationConfig.response_format)),
            output_format=str(image_cfg.get("output_format", ImageGenerationConfig.output_format)),
            api_key_env=str(image_cfg.get("api_key_env", ImageGenerationConfig.api_key_env)),
            api_key=image_cfg.get("api_key", ImageGenerationConfig.api_key),
            timeout_seconds=int(image_cfg.get("timeout_seconds", ImageGenerationConfig.timeout_seconds)),
            default_images=tuple(str(item) for item in default_images),
            default_mask=(
                str(image_cfg.get("default_mask"))
                if image_cfg.get("default_mask") not in (None, "")
                else ImageGenerationConfig.default_mask
            ),
        )

    def _resolve_api_key(self) -> str | None:
        if self._config.api_key:
            return self._config.api_key
        return os.environ.get(self._config.api_key_env)

    @staticmethod
    def _extract_url(payload: Any) -> str:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and isinstance(first.get("url"), str):
                    return first["url"]

            if isinstance(payload.get("url"), str):
                return payload["url"]

            result = payload.get("result")
            if isinstance(result, dict) and isinstance(result.get("url"), str):
                return result["url"]

        return ""

    def generate(
        self,
        image_prompt: str,
        *,
        image_paths: Sequence[str | Path] | None = None,
        mask_path: str | Path | None = None,
    ) -> str:
        cleaned_prompt = image_prompt.strip()
        if not cleaned_prompt:
            return ""

        if self._config.provider != "imagerouter":
            return ""

        api_key = self._resolve_api_key()
        if not api_key:
            return ""

        effective_images = list(image_paths or self._config.default_images)
        if not effective_images:
            return ""

        effective_mask = mask_path or self._config.default_mask

        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "prompt": cleaned_prompt,
            "model": self._config.model,
            "quality": self._config.quality,
            "size": self._config.size,
            "response_format": self._config.response_format,
            "output_format": self._config.output_format,
        }

        open_handles: list[Any] = []
        files: list[tuple[str, tuple[str, Any, str]]] = []

        try:
            for raw_path in effective_images:
                path = Path(raw_path)
                if not path.exists() or not path.is_file():
                    continue
                handle = path.open("rb")
                open_handles.append(handle)
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("image[]", (path.name, handle, mime_type)))

            if effective_mask:
                mask = Path(effective_mask)
                if mask.exists() and mask.is_file():
                    handle = mask.open("rb")
                    open_handles.append(handle)
                    mime_type = mimetypes.guess_type(mask.name)[0] or "application/octet-stream"
                    files.append(("mask[]", (mask.name, handle, mime_type)))

            if not files:
                return ""

            response = requests.post(
                self._config.endpoint,
                files=files,
                data=payload,
                headers=headers,
                timeout=self._config.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            return self._extract_url(body)
        except requests.RequestException:
            return ""
        except ValueError:
            return ""
        finally:
            for handle in open_handles:
                try:
                    handle.close()
                except OSError:
                    pass