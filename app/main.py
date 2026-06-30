from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.serving import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = state.load()
    print("Recommender loaded" if loaded else "No artifacts yet — run the training pipeline.")
    yield


app = FastAPI(title="Mentra ML Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.post("/reload", tags=["admin"])
def reload_model() -> dict:
    """Hot-reload artifacts after a retrain (the slow-loop model swap)."""
    loaded = state.load()
    return {"reloaded": loaded}
