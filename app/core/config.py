from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Reads the same Postgres the backend owns. The ML service only READS core
    # tables and writes only its own artifacts (never business tables).
    DATABASE_URL: str = "postgresql+psycopg://postgres:admin@localhost:5432/elearning"

    ARTIFACT_DIR: str = "artifacts"

    # --- Content model ---
    TFIDF_MAX_FEATURES: int = 30000
    TFIDF_MIN_DF: int = 2

    # --- Collaborative filtering (SVD latent-factor MF over implicit feedback) ---
    CF_FACTORS: int = 48

    # Implicit-feedback confidence weights per signal (summed per user/course).
    W_ENROLL: float = 1.0
    W_COMPLETION: float = 2.0   # multiplied by completion fraction
    W_REVIEW: float = 1.5       # multiplied by rating/5
    W_WISHLIST: float = 0.5

    # --- Ensemble blend (used when the ranker is absent) ---
    # Tuned by the RQ1 method ablation: on this catalog the content signal is the
    # strongest single method and CF is sparser, so the blend leans on content
    # while still folding in collaborative + popularity. This makes the hybrid
    # meet-or-beat every single method it combines (see pipelines/research.py).
    BLEND_CF: float = 0.25
    BLEND_CONTENT: float = 0.65
    BLEND_POPULARITY: float = 0.10

    # --- Candidate generation ---
    CAND_PER_SOURCE: int = 100
    DEFAULT_K: int = 10


settings = Settings()


def artifact_dir() -> Path:
    p = Path(settings.ARTIFACT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p
