from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _normalize_origin(value: str) -> str:
    return value.strip().rstrip("/")


def _parse_cors_allow_origins() -> tuple[str, ...]:
    """Merge comma-separated UI_ALLOWED_ORIGINS, legacy UI_ALLOWED_ORIGIN, and local dev defaults."""
    origins: list[str] = []
    multi = os.getenv("UI_ALLOWED_ORIGINS", "").strip()
    if multi:
        origins.extend(_normalize_origin(x) for x in multi.split(",") if x.strip())
    single = _normalize_origin(os.getenv("UI_ALLOWED_ORIGIN", "http://localhost:5173"))
    if single not in origins:
        origins.append(single)
    for dev in ("http://127.0.0.1:5173", "http://localhost:5173", "http://localhost:3000"):
        if dev not in origins:
            origins.append(dev)
    seen: set[str] = set()
    unique: list[str] = []
    for o in origins:
        if o not in seen:
            seen.add(o)
            unique.append(o)
    return tuple(unique)


# Evaluated once at import (restart server after changing env).
CORS_ALLOW_ORIGINS: tuple[str, ...] = _parse_cors_allow_origins()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    output_root: Path = PROJECT_ROOT / "outputs"
    upload_root: Path = PROJECT_ROOT / "data" / "uploads"
    local_store_root: Path = PROJECT_ROOT / ".localdata"
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    supabase_bucket: str = os.getenv("SUPABASE_BUCKET", "quiz-assets")
    allowed_origin: str = _normalize_origin(os.getenv("UI_ALLOWED_ORIGIN", "http://localhost:5173"))

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def auth_enabled(self) -> bool:
        """When True, notebook APIs require a valid Supabase user JWT and scope data by owner_id.

        Uses project JWKS (asymmetric keys, ES256/RS256) when the token header says so; otherwise
        HS256 + SUPABASE_JWT_SECRET for legacy signing keys.
        """
        return bool(self.supabase_url)


settings = Settings()
