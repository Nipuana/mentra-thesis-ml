from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.schemas import HealthResponse, RecResponse
from app.core.auth import require_token
from app.pipelines import evaluate as evaluate_mod
from app.pipelines import research as research_mod
from app.serving import state

# `/health` stays open so process supervisors and container health checks can
# probe it without a credential; everything that exposes model output or data
# requires the shared token.
health_router = APIRouter()
router = APIRouter(dependencies=[Depends(require_token)])

# Offline evaluation is a few seconds to compute; cache it in-process.
_METRICS_CACHE: dict | None = None
_RQ1_CACHE: dict | None = None
_RQ2_CACHE: dict | None = None


def _require_ready():
    if state.recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m app.pipelines.train_pipeline` first.",
        )
    return state.recommender


@health_router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    ready = state.recommender is not None
    manifest = state.recommender.manifest if ready else {}
    return HealthResponse(status="ok", ready=ready, manifest=manifest)


@router.get("/recommend/{user_id}", response_model=RecResponse, tags=["recommend"])
def recommend(user_id: str, k: int = Query(10, ge=1, le=50)) -> RecResponse:
    rec = _require_ready()
    items = rec.recommend(user_id, k)
    strategy = "personalized" if rec.cf.knows_user(user_id) or user_id in rec.user_profiles else "cold_start_popularity"
    return RecResponse(items=items, strategy=strategy, model_version=rec.manifest.get("trained_at"))


@router.get("/similar/{course_id}", response_model=RecResponse, tags=["recommend"])
def similar(course_id: str, k: int = Query(10, ge=1, le=50)) -> RecResponse:
    rec = _require_ready()
    if course_id not in rec.meta:
        raise HTTPException(status_code=404, detail="Course not found")
    return RecResponse(items=rec.similar(course_id, k), strategy="content_similarity",
                       model_version=rec.manifest.get("trained_at"))


@router.get("/search", response_model=RecResponse, tags=["search"])
def search(q: str = Query(..., min_length=1), k: int = Query(20, ge=1, le=50)) -> RecResponse:
    rec = _require_ready()
    return RecResponse(items=rec.search(q, k), strategy="hybrid_search",
                       model_version=rec.manifest.get("trained_at"))


@router.get("/trending", response_model=RecResponse, tags=["recommend"])
def trending(k: int = Query(10, ge=1, le=50)) -> RecResponse:
    rec = _require_ready()
    return RecResponse(items=rec.trending(k), strategy="popularity",
                       model_version=rec.manifest.get("trained_at"))


@router.get("/sectors", tags=["admin"])
def sectors() -> dict:
    rec = _require_ready()
    return {"sectors": rec.sectors()}


@router.get("/metrics", tags=["admin"])
def metrics(refresh: bool = Query(False)) -> dict:
    """Offline evaluation: hybrid recommender vs popularity baseline
    (Recall/Precision/NDCG@K + Coverage). Cached after first run."""
    global _METRICS_CACHE
    if _METRICS_CACHE is None or refresh:
        res = evaluate_mod.run()
        res["generated_at"] = datetime.now(timezone.utc).isoformat()
        _METRICS_CACHE = res
    return _METRICS_CACHE


@router.get("/research/rq1", tags=["research"])
def research_rq1(refresh: bool = Query(False)) -> dict:
    """RQ1 evidence: method ablation — Popularity vs Content-based vs
    Collaborative vs the Hybrid blend, on the same leave-out split."""
    global _RQ1_CACHE
    if _RQ1_CACHE is None or refresh:
        res = research_mod.rq1()
        res["generated_at"] = datetime.now(timezone.utc).isoformat()
        _RQ1_CACHE = res
    return _RQ1_CACHE


@router.get("/research/rq2", tags=["research"])
def research_rq2(refresh: bool = Query(False)) -> dict:
    """RQ2 evidence: how well anonymized interaction signals (view frequency,
    time-on-lesson, completion) proxy learning preference and difficulty fit."""
    global _RQ2_CACHE
    if _RQ2_CACHE is None or refresh:
        res = research_mod.rq2()
        res["generated_at"] = datetime.now(timezone.utc).isoformat()
        _RQ2_CACHE = res
    return _RQ2_CACHE


@router.get("/simulate", tags=["admin"])
def simulate(fields: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=30)) -> dict:
    """Cold-start what-if: recommend for a brand-new student who declared one or
    more specialized interests (comma-separated `fields`, e.g. "React,Rust")."""
    rec = _require_ready()
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    res = rec.simulate_new_student(field_list, k)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Unknown field(s) '{fields}'")
    return {"k": k, "strategy": "cold_start_content", **res}
