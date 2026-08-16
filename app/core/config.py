from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Reads the same Postgres the backend owns. The ML service only READS core
    # tables and writes only its own artifacts (never business tables).
    DATABASE_URL: str = "postgresql+psycopg://postgres:admin@localhost:5432/elearning"

    ARTIFACT_DIR: str = "artifacts"

    # --- Service auth ---
    # This service is called server-to-server by the backend only; it is never
    # reached from a browser. Requests must carry `X-ML-Token` matching this
    # value. Empty means the check is disabled (local development only).
    API_TOKEN: str = ""

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
    # Selected by grid search on the VALIDATION block only
    # (`python -m app.pipelines.evaluate tune`), never on the test block.
    #
    # These weights replace an earlier content-heavy setting (.25/.65/.10) that
    # was chosen under a random per-user split. Once the split became temporal —
    # train strictly on a learner's past, predict their future — the ordering
    # reversed: collaborative signal is the strongest single method and content
    # is the weaker one, because a random split let the content model match
    # against courses the learner had already taken. Validation NDCG@10 is a
    # broad plateau over cf .55-.75, so the exact point is not load-bearing.
    BLEND_CF: float = 0.65
    BLEND_CONTENT: float = 0.25
    BLEND_POPULARITY: float = 0.10

    # --- Candidate generation ---
    CAND_PER_SOURCE: int = 100
    DEFAULT_K: int = 10

    # --- Which scorer serves live traffic ---
    # False = the weighted blend above; True = the LightGBM ranker.
    #
    # Evaluated head-to-head on the temporal split (`evaluate ablation`): the two
    # are statistically indistinguishable on accuracy (NDCG@10 .261 vs .254,
    # Wilcoxon p = .12), but the blend covers 32% of the catalogue against the
    # ranker's 19% and puts 12% of its recommendations outside the popular head
    # against the ranker's 0%. Equal accuracy, strictly wider reach, simpler
    # model — so the blend serves and the ranker stays available behind this
    # flag. Flip it to re-deploy the ranker; the ablation reports both either way.
    USE_RANKER: bool = False


settings = Settings()


def artifact_dir() -> Path:
    p = Path(settings.ARTIFACT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p
