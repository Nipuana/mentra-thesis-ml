# Mentra — ML / Recommendation Service

A separate, independently-deployable FastAPI service that serves recommendations,
"similar courses", and semantic-ish search. Reads the same Postgres the backend
owns (read-only on business tables; writes only its own artifacts).

## Approach (and production upgrade path)

| Layer | Built with (runs now) | Production upgrade |
|---|---|---|
| Content-based | TF-IDF (1–2gram) over transcripts + title + description + taxonomy | sentence-transformers embeddings |
| Collaborative filtering | confidence-weighted implicit matrix → `TruncatedSVD` latent factors | `implicit` ALS |
| Ranking | LightGBM over blended features (CF, content, popularity, rating, price, difficulty, duration) | LambdaRank + richer features |
| Candidate blend fallback | weighted ensemble (CF/content/popularity) | — |
| Vector store / registry | numpy + joblib artifacts | pgvector / FAISS + MLflow |
| Serving | FastAPI on :8100, artifacts loaded at startup, `/reload` for hot-swap | + Redis cache |

The interfaces (vectorizer+matrix, user/item factors) are drop-in seams for the
heavier libraries.

## Setup & run

Postgres must be seeded (see `../backend`). Python 3.13 venv:

```bash
cd ml-service
py -3.13 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 1) train — builds content vectors, CF factors, LightGBM ranker (~seconds)
./.venv/Scripts/python.exe -m app.pipelines.train_pipeline

# 2) evaluate (RQ1) — hybrid vs popularity baseline
./.venv/Scripts/python.exe -m app.pipelines.evaluate

# 3) serve
./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8100
```

## Endpoints (docs at http://localhost:8100/docs)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | readiness + training manifest |
| GET | `/recommend/{user_id}?k=10` | hybrid personalized; cold-start → popularity; excludes already-engaged |
| GET | `/similar/{course_id}?k=10` | content similarity |
| GET | `/search?q=…&k=20` | TF-IDF query over course vectors |
| GET | `/trending?k=10` | popularity |
| POST | `/reload` | hot-reload artifacts after a retrain |

Responses return full course metadata (title, sector, thumbnail, rating, price,
`score`, `reason`) so the frontend can render results directly.

## Data contract
Reads `courses`, `lessons` (transcripts), `categories`, `enrollments`, `reviews`,
`wishlist_items`. CF signals come from the **account-bound** tables (which carry
`user_id` — the user's own data, used to serve them). The anonymized
`interaction_events` log is available for future aggregate/session models.

## Latest results (offline eval, RQ1)

| Metric | Popularity | Hybrid | Lift |
|---|---|---|---|
| Recall@10 | 0.013 | 0.144 | +1008% |
| Precision@10 | 0.004 | 0.040 | +1022% |
| NDCG@10 | 0.006 | 0.090 | +1380% |
| Coverage | 0.012 | 0.511 | +4158% |

> The strong lift reflects the seed's built-in latent taste being learnable; it
> validates the pipeline and demonstrates personalization ≫ popularity. Real
> uplift numbers will come from live interaction data (frontend tracking).

## Structure
```
app/
  main.py                 FastAPI + lifespan (loads artifacts) + /reload
  core/       config.py (weights, hyperparams), db.py
  data/       loaders.py  (SQL → DataFrames)
  models/     content_based.py, collaborative.py, ranker.py
  serving/    recommender.py (candidate gen + blend/rank + rules), state.py
  pipelines/  train_pipeline.py, evaluate.py
  registry/   store.py (joblib artifacts + manifest)
  api/        routes.py, schemas.py
artifacts/                trained models (gitignored)
```
