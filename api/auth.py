from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings

security = HTTPBearer(auto_error=False)

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        url = f"{settings.supabase_url}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, cache_keys=True)
    return _jwks_client


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: str | None


def _decode_supabase_jwt(token: str) -> AuthUser:
    issuer = f"{settings.supabase_url}/auth/v1"
    header = jwt.get_unverified_header(token)
    token_alg = header.get("alg")

    try:
        if token_alg in ("ES256", "RS256"):
            signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=[token_alg],
                audience="authenticated",
                issuer=issuer,
            )
        elif token_alg == "HS256":
            if not settings.supabase_jwt_secret:
                raise HTTPException(
                    status_code=503,
                    detail="SUPABASE_JWT_SECRET is required for HS256 (legacy) access tokens.",
                )
            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=issuer,
            )
        else:
            raise jwt.InvalidAlgorithmError(f"Unsupported JWT alg: {token_alg!r}")
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc

    sub = payload.get("sub")
    if not sub or not isinstance(sub, str):
        raise HTTPException(status_code=401, detail="Invalid session")
    email = payload.get("email")
    return AuthUser(id=sub, email=email if isinstance(email, str) else None)


async def get_auth_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthUser | None:
    if not settings.auth_enabled:
        return None
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _decode_supabase_jwt(credentials.credentials)
