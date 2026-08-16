from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.routes import health_router, router
from app.core.auth import require_token
from app.serving import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = state.load()
    print("Recommender loaded" if loaded else "No artifacts yet — run the training pipeline.")
    yield


app = FastAPI(title="Mentra ML Service", version="0.1.0", lifespan=lifespan)

# No CORS middleware by design: this service is called server-to-server by the
# backend and is never a browser origin. Bind it to localhost (or a private
# network) and keep API_TOKEN set outside local development.

app.include_router(health_router)
app.include_router(router)


@app.post("/reload", tags=["admin"], dependencies=[Depends(require_token)])
def reload_model() -> dict:
    """Hot-reload artifacts after a retrain (the slow-loop model swap).

    NOTE: the recommender lives in process memory, so this reloads only the
    worker that serves the request. Run this service with a single worker, or
    reload every worker, after a retrain.
    """
    loaded = state.load()
    return {"reloaded": loaded}
