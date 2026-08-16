"""Shared-secret auth for the ML service.

The service reads the production database and exposes the trained model, but is
only ever called server-to-server by the backend. A shared token on `X-ML-Token`
keeps it from being driven directly by anything that can reach the port.

Set `API_TOKEN` here and `ML_API_TOKEN` in the backend to the same value. When
both are empty the check is skipped, which is the local-development default.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_token(x_ml_token: str | None = Header(default=None)) -> None:
    if not settings.API_TOKEN:
        return  # auth disabled (local development)
    if not x_ml_token or not secrets.compare_digest(x_ml_token, settings.API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-ML-Token",
        )
