from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from fastapi import HTTPException

from .config import Settings


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _rest_fail(response: requests.Response) -> None:
    """Raise HTTPException with PostgREST / Supabase error body (no secrets)."""
    detail: str
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(
                body.get("message")
                or body.get("error_description")
                or body.get("hint")
                or body.get("code")
                or json.dumps(body, ensure_ascii=False)
            )
        else:
            detail = str(body)
    except Exception:
        detail = (response.text or response.reason or "Supabase REST error").strip()
    status = response.status_code
    if status >= 500:
        status = 502
    raise HTTPException(status_code=status, detail=detail[:4000])


class NotebookRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.local_store_root.mkdir(parents=True, exist_ok=True)
        self._local_files = {
            "notebooks": self.settings.local_store_root / "notebooks.json",
            "sources": self.settings.local_store_root / "sources.json",
            "messages": self.settings.local_store_root / "messages.json",
            "runs": self.settings.local_store_root / "runs.json",
        }

    @property
    def is_remote(self) -> bool:
        return self.settings.supabase_enabled

    def list_notebooks(self, *, owner_id: str | None = None) -> list[dict[str, Any]]:
        notebooks = self._select("notebooks", order="last_opened_at.desc")
        if owner_id is not None:
            notebooks = [item for item in notebooks if item.get("owner_id") == owner_id]
        return sorted(notebooks, key=lambda item: item.get("last_opened_at", ""), reverse=True)

    def get_notebook(self, notebook_id: str) -> dict[str, Any] | None:
        rows = self._select("notebooks", filters={"id": notebook_id})
        return rows[0] if rows else None

    def create_notebook(
        self,
        title: str,
        description: str | None = None,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        now = utcnow_iso()
        payload: dict[str, Any] = {
            "id": str(uuid4()),
            "title": title,
            "description": description or "",
            "cover_image": "",
            "accent_color": self._pick_accent(title),
            "pinned": False,
        }
        if not self.is_remote:
            payload["created_at"] = now
            payload["updated_at"] = now
            payload["last_opened_at"] = now
        if owner_id:
            payload["owner_id"] = owner_id
        return self._insert_one("notebooks", payload)

    def update_notebook(self, notebook_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        patch = {**patch, "updated_at": utcnow_iso()}
        rows = self._update("notebooks", filters={"id": notebook_id}, payload=patch)
        return rows[0] if rows else None

    def list_sources(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._select("notebook_sources", filters={"notebook_id": notebook_id}, order="created_at.asc")

    def create_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = {
            "id": str(uuid4()),
            "created_at": utcnow_iso(),
            **payload,
        }
        return self._insert_one("notebook_sources", source)

    def list_messages(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._select("notebook_messages", filters={"notebook_id": notebook_id}, order="created_at.asc")

    def create_message(
        self,
        notebook_id: str,
        role: str,
        content: str,
        *,
        kind: str = "chat",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "id": str(uuid4()),
            "notebook_id": notebook_id,
            "role": role,
            "content": content,
            "kind": kind,
            "metadata": metadata or {},
            "created_at": utcnow_iso(),
        }
        return self._insert_one("notebook_messages", payload)

    def list_runs(self, notebook_id: str) -> list[dict[str, Any]]:
        return self._select("notebook_runs", filters={"notebook_id": notebook_id}, order="created_at.desc")

    def get_run_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        rows = self._select("notebook_runs", filters={"run_id": run_id})
        return rows[0] if rows else None

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = utcnow_iso()
        run = {
            "id": str(uuid4()),
            "created_at": now,
            "updated_at": now,
            **payload,
        }
        return self._insert_one("notebook_runs", run)

    def update_run(self, run_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        patch = {**patch, "updated_at": utcnow_iso()}
        rows = self._update("notebook_runs", filters={"run_id": run_id}, payload=patch)
        return rows[0] if rows else None

    def delete_run(self, run_id: str) -> None:
        self._delete("notebook_runs", filters={"run_id": run_id})

    def upload_blob(self, storage_path: str, payload: bytes, content_type: str) -> str:
        if self.is_remote:
            url = f"{self.settings.supabase_url}/storage/v1/object/{self.settings.supabase_bucket}/{storage_path}"
            headers = {
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
                "x-upsert": "true",
                "content-type": content_type,
            }
            response = requests.post(url, headers=headers, data=payload, timeout=30)
            if not response.ok:
                _rest_fail(response)
            return (
                f"{self.settings.supabase_url}/storage/v1/object/public/"
                f"{self.settings.supabase_bucket}/{storage_path}"
            )

        local_target = self.settings.local_store_root / "storage" / storage_path
        local_target.parent.mkdir(parents=True, exist_ok=True)
        local_target.write_bytes(payload)
        relative = local_target.relative_to(self.settings.project_root).as_posix()
        return f"/api/artifacts/{relative}"

    def _pick_accent(self, title: str) -> str:
        palette = ["#5f6fff", "#4fb38a", "#e37a46", "#9d6bff", "#f0bf57"]
        return palette[sum(ord(char) for char in title) % len(palette)]

    def _select(
        self,
        table: str,
        *,
        filters: dict[str, Any] | None = None,
        order: str | None = None,
    ) -> list[dict[str, Any]]:
        if self.is_remote:
            params: dict[str, Any] = {"select": "*"}
            if order:
                params["order"] = order
            for key, value in (filters or {}).items():
                params[key] = f"eq.{value}"
            response = requests.get(
                f"{self.settings.supabase_url}/rest/v1/{table}",
                headers=self._rest_headers(),
                params=params,
                timeout=30,
            )
            if not response.ok:
                _rest_fail(response)
            return response.json()

        items = self._read_local(table)
        filtered = [
            item for item in items if all(item.get(key) == value for key, value in (filters or {}).items())
        ]
        if order:
            field, direction = order.split(".")
            reverse = direction == "desc"
            filtered.sort(key=lambda item: item.get(field, ""), reverse=reverse)
        return filtered

    def _insert_one(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_remote:
            response = requests.post(
                f"{self.settings.supabase_url}/rest/v1/{table}",
                headers={**self._rest_headers(), "Prefer": "return=representation"},
                json=payload,
                timeout=30,
            )
            if not response.ok:
                _rest_fail(response)
            return response.json()[0]

        items = self._read_local(table)
        items.append(payload)
        self._write_local(table, items)
        return payload

    def _update(
        self,
        table: str,
        *,
        filters: dict[str, Any],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.is_remote:
            params = {key: f"eq.{value}" for key, value in filters.items()}
            response = requests.patch(
                f"{self.settings.supabase_url}/rest/v1/{table}",
                headers={**self._rest_headers(), "Prefer": "return=representation"},
                params=params,
                json=payload,
                timeout=30,
            )
            if not response.ok:
                _rest_fail(response)
            return response.json()

        items = self._read_local(table)
        updated: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if all(item.get(key) == value for key, value in filters.items()):
                items[index] = {**item, **payload}
                updated.append(items[index])
        self._write_local(table, items)
        return updated

    def _delete(self, table: str, *, filters: dict[str, Any]) -> None:
        if self.is_remote:
            params = {key: f"eq.{value}" for key, value in filters.items()}
            response = requests.delete(
                f"{self.settings.supabase_url}/rest/v1/{table}",
                headers=self._rest_headers(),
                params=params,
                timeout=30,
            )
            if not response.ok:
                _rest_fail(response)
            return

        items = self._read_local(table)
        remaining = [item for item in items if not all(item.get(key) == value for key, value in filters.items())]
        self._write_local(table, remaining)

    def _read_local(self, table: str) -> list[dict[str, Any]]:
        path = self._local_files[self._normalize_local_table(table)]
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_local(self, table: str, items: list[dict[str, Any]]) -> None:
        path = self._local_files[self._normalize_local_table(table)]
        path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    def _normalize_local_table(self, table: str) -> str:
        mapping = {
            "notebooks": "notebooks",
            "notebook_sources": "sources",
            "notebook_messages": "messages",
            "notebook_runs": "runs",
        }
        return mapping[table]

    def _rest_headers(self) -> dict[str, str]:
        return {
            "apikey": self.settings.supabase_service_role_key,
            "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
            "content-type": "application/json",
        }
