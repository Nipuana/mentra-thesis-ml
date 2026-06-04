"""Holds the loaded Recommender so routes and the lifespan share one instance
(enables hot-reload after a retrain without restarting the process)."""

from app.registry import store

recommender = None  # type: ignore[assignment]


def load() -> bool:
    global recommender
    if store.exists("content") and store.exists("cf") and store.exists("courses"):
        from app.serving.recommender import Recommender

        recommender = Recommender()
        return True
    recommender = None
    return False
