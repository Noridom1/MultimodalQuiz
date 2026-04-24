from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import requests
import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"freepik", "freepik_gemini", "freepik/gemini", "google", "genai"}


@dataclass(frozen=True)
class ImageGenerationConfig:
    provider: str = "freepik"
    endpoint: str = "https://api.freepik.com/v1/ai/gemini-2-5-flash-image-preview"
    model: str = "gemini-2-5-flash-image-preview"
    output_format: str = "png"
    api_key_env: str = "FREEPIK_API_KEY"
    api_key: str | None = None
    timeout_seconds: int = 120
    poll_interval_seconds: int = 2


class ImageGenerator:
    def __init__(self, pipeline=None, *, config_path: str | Path | None = None):
        self.pipeline = pipeline
        self._config = self._load_config(config_path)
        LOGGER.info(
            "Initialized image generator provider=%s model=%s endpoint=%s api_key_env=%s",
            self._config.provider,
            self._config.model,
            self._config.endpoint,
            self._config.api_key_env,
        )

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

        return ImageGenerationConfig(
            provider=str(image_cfg.get("provider", ImageGenerationConfig.provider)),
            endpoint=str(image_cfg.get("endpoint", ImageGenerationConfig.endpoint)),
            model=str(image_cfg.get("model", ImageGenerationConfig.model)),
            output_format=str(image_cfg.get("output_format", ImageGenerationConfig.output_format)),
            api_key_env=str(image_cfg.get("api_key_env", ImageGenerationConfig.api_key_env)),
            api_key=image_cfg.get("api_key", ImageGenerationConfig.api_key),
            timeout_seconds=int(image_cfg.get("timeout_seconds", ImageGenerationConfig.timeout_seconds)),
            poll_interval_seconds=max(
                1, int(image_cfg.get("poll_interval_seconds", ImageGenerationConfig.poll_interval_seconds))
            ),
        )

    def _resolve_api_key(self) -> str | None:
        if self._config.api_key:
            LOGGER.info("Using image generation API key from config for provider=%s", self._config.provider)
            return self._config.api_key
        if load_dotenv is not None:
            try:
                project_root = Path(__file__).resolve().parents[2]
                load_dotenv(project_root / ".env")
            except Exception:
                pass
        api_key = os.environ.get(self._config.api_key_env)
        if api_key:
            LOGGER.info("Resolved image generation API key from env var=%s", self._config.api_key_env)
        else:
            LOGGER.warning("Image generation API key not found in env var=%s", self._config.api_key_env)
        return api_key

    def _output_dir(self, output_dir: str | Path | None = None) -> Path:
        out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[2] / "data" / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _download_to_file(
        self,
        url: str,
        *,
        output_dir: str | Path | None = None,
        file_stem: str | None = None,
    ) -> str:
        LOGGER.info("Downloading generated image from %s", url)
        response = requests.get(url, timeout=self._config.timeout_seconds)
        response.raise_for_status()

        ext = (self._config.output_format or "").lstrip(".")
        if not ext:
            content_type = response.headers.get("Content-Type", "")
            ext = content_type.split("/")[-1] if "/" in content_type else "png"

        stem = file_stem or f"img_{uuid4().hex}"
        dest = self._output_dir(output_dir) / f"{stem}.{ext}"
        dest.write_bytes(response.content)
        LOGGER.info("Saved generated image to %s", dest)
        return str(dest)

    @staticmethod
    def _is_public_url(value: str) -> bool:
        lowered = value.lower()
        return lowered.startswith("http://") or lowered.startswith("https://")

    def _encode_reference_image(self, path_or_url: str | Path) -> str:
        raw_value = str(path_or_url)
        if self._is_public_url(raw_value):
            return raw_value

        path = Path(path_or_url)
        with path.open("rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")

    def _build_reference_images(self, image_paths: Sequence[str | Path] | None, mask_path: str | Path | None) -> list[str]:
        refs: list[str] = []
        for item in image_paths or ():
            if len(refs) >= 3:
                break
            try:
                refs.append(self._encode_reference_image(item))
            except OSError:
                LOGGER.debug("Skipping unreadable reference image: %s", item, exc_info=True)

        if mask_path:
            LOGGER.debug("Ignoring mask_path because Freepik Gemini endpoint only supports reference_images")

        return refs

    def _submit_task(self, *, prompt: str, reference_images: list[str], api_key: str) -> dict[str, Any]:
        headers = {
            "x-freepik-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"prompt": prompt}
        if reference_images:
            payload["reference_images"] = reference_images

        response = requests.post(
            self._config.endpoint,
            json=payload,
            headers=headers,
            timeout=self._config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _get_task(self, task_id: str, api_key: str) -> dict[str, Any]:
        headers = {"x-freepik-api-key": api_key}
        response = requests.get(
            f"{self._config.endpoint}/{task_id}",
            headers=headers,
            timeout=self._config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_task_data(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data", {})
        return data if isinstance(data, dict) else {}

    def generate(
        self,
        image_prompt: str,
        *,
        image_paths: Sequence[str | Path] | None = None,
        mask_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        file_stem: str | None = None,
    ) -> str:
        cleaned_prompt = image_prompt.strip()
        started_at = time.perf_counter()
        LOGGER.info(
            "Starting image generation provider=%s model=%s prompt_len=%d reference_images=%d output_dir=%s file_stem=%s",
            self._config.provider,
            self._config.model,
            len(cleaned_prompt),
            len(image_paths or []),
            output_dir,
            file_stem,
        )
        if not cleaned_prompt:
            LOGGER.warning("Empty image prompt provided; aborting image generation")
            return ""

        provider = (self._config.provider or "").lower()
        if provider not in SUPPORTED_PROVIDERS:
            LOGGER.error("Unsupported image provider=%s", self._config.provider)
            return ""

        api_key = self._resolve_api_key()
        if not api_key:
            LOGGER.error("No API key found (env var=%s); aborting image generation", self._config.api_key_env)
            return ""

        reference_images = self._build_reference_images(image_paths, mask_path)
        LOGGER.info(
            "Prepared image request endpoint=%s model=%s references=%d",
            self._config.endpoint,
            self._config.model,
            len(reference_images),
        )

        try:
            LOGGER.info("Submitting image generation task to provider=%s", self._config.provider)
            create_payload = self._submit_task(prompt=cleaned_prompt, reference_images=reference_images, api_key=api_key)
            task_data = self._extract_task_data(create_payload)
        except Exception:
            LOGGER.exception("Image generation task submission failed provider=%s", self._config.provider)
            return ""

        task_id = str(task_data.get("task_id", "")).strip()
        status = str(task_data.get("status", "")).strip().upper()
        generated = task_data.get("generated") if isinstance(task_data.get("generated"), list) else []

        if not task_id:
            LOGGER.error("Image generation response did not include task_id provider=%s payload=%s", self._config.provider, create_payload)
            return ""

        LOGGER.info("Submitted image generation task task_id=%s initial_status=%s", task_id, status or "UNKNOWN")

        deadline = time.monotonic() + self._config.timeout_seconds
        while status in {"", "CREATED", "IN_PROGRESS"} and time.monotonic() < deadline:
            time.sleep(self._config.poll_interval_seconds)
            try:
                poll_payload = self._get_task(task_id, api_key)
                task_data = self._extract_task_data(poll_payload)
            except Exception:
                LOGGER.exception("Image generation task polling failed task_id=%s", task_id)
                return ""

            status = str(task_data.get("status", "")).strip().upper()
            generated = task_data.get("generated") if isinstance(task_data.get("generated"), list) else []
            LOGGER.info("Polled image generation task task_id=%s status=%s", task_id, status or "UNKNOWN")

            if status == "FAILED":
                LOGGER.error("Image generation task failed task_id=%s payload=%s", task_id, poll_payload)
                return ""
            if status == "COMPLETED":
                break

        if status != "COMPLETED":
            LOGGER.error("Image generation timed out waiting for completion task_id=%s status=%s", task_id, status)
            return ""

        image_url = next((str(item).strip() for item in generated if str(item).strip()), "")
        if not image_url:
            LOGGER.error("Image generation completed without generated image URL task_id=%s payload=%s", task_id, task_data)
            return ""

        try:
            local_path = self._download_to_file(
                image_url,
                output_dir=output_dir,
                file_stem=file_stem,
            )
            LOGGER.info(
                "Completed image generation task_id=%s local_path=%s elapsed=%.2fs",
                task_id,
                local_path,
                time.perf_counter() - started_at,
            )
            return local_path
        except Exception:
            LOGGER.exception("Failed to download generated image task_id=%s; returning source URL instead", task_id)
            LOGGER.info(
                "Completed image generation task_id=%s source_url=%s elapsed=%.2fs",
                task_id,
                image_url,
                time.perf_counter() - started_at,
            )
            return image_url
