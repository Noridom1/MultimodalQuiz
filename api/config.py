from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    output_root: Path = PROJECT_ROOT / "outputs"
    upload_root: Path = PROJECT_ROOT / "data" / "uploads"
    local_store_root: Path = PROJECT_ROOT / ".localdata"
    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_bucket: str = os.getenv("SUPABASE_BUCKET", "quiz-assets")
    allowed_origin: str = os.getenv("UI_ALLOWED_ORIGIN", "http://localhost:5173")

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)


settings = Settings()

