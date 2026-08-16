"""The ML service reads the production database, so the token check is the only
thing standing between an open port and the model + data behind it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app

TOKEN = "test-token"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(config.settings, "API_TOKEN", TOKEN)
    return TestClient(app)


@pytest.fixture
def open_client(monkeypatch) -> TestClient:
    """Auth disabled — the documented local-development default."""
    monkeypatch.setattr(config.settings, "API_TOKEN", "")
    return TestClient(app)


def test_health_needs_no_token(client):
    """Container health checks must work without a credential."""
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("path", ["/trending", "/search?q=python", "/sectors", "/metrics"])
def test_protected_routes_reject_missing_token(client, path):
    assert client.get(path).status_code == 401


def test_protected_route_rejects_wrong_token(client):
    assert client.get("/trending", headers={"X-ML-Token": "nope"}).status_code == 401


def test_reload_requires_a_token(client):
    """Reload swaps the model that serves every learner."""
    assert client.post("/reload").status_code == 401


def test_correct_token_passes_the_check(client):
    # 503 = "no model loaded in this process", which is past the auth layer.
    assert client.get("/trending", headers={"X-ML-Token": TOKEN}).status_code in (200, 503)


def test_empty_token_setting_disables_the_check(open_client):
    assert open_client.get("/trending").status_code in (200, 503)
