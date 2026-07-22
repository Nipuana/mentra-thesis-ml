"""Artifact registry. Joblib files on disk today; the same save/load surface is
where an MLflow model registry would plug in later.
"""

import json
from datetime import datetime, timezone
from typing import Any

import joblib

from app.core.config import artifact_dir

MANIFEST = "manifest.json"


def save(name: str, obj: Any) -> None:
    joblib.dump(obj, artifact_dir() / f"{name}.joblib")


def load(name: str) -> Any:
    return joblib.load(artifact_dir() / f"{name}.joblib")


def exists(name: str) -> bool:
    return (artifact_dir() / f"{name}.joblib").exists()


def write_manifest(meta: dict) -> None:
    meta = {**meta, "trained_at": datetime.now(timezone.utc).isoformat()}
    (artifact_dir() / MANIFEST).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_manifest() -> dict:
    path = artifact_dir() / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
