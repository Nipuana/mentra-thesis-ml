"""Collaborative filtering via latent-factor matrix factorization.

Builds a confidence-weighted user x item implicit-feedback matrix and factorizes
it with TruncatedSVD to get user/item factors. The (user_factors, item_factors)
interface matches `implicit` ALS, so ALS can be swapped in later without touching
callers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from app.core.config import settings


class CFModel:
    def __init__(self) -> None:
        self.user_ids: list[str] = []
        self.item_ids: list[str] = []
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame, valid_items: set[str] | None = None) -> "CFModel":
        df = interactions
        if valid_items is not None:
            df = df[df["course_id"].isin(valid_items)]
        df = df[df["weight"] > 0]

        self.user_ids = sorted(df["user_id"].unique().tolist())
        self.item_ids = sorted(df["course_id"].unique().tolist())
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {c: i for i, c in enumerate(self.item_ids)}

        rows = df["user_id"].map(self.user_index).to_numpy()
        cols = df["course_id"].map(self.item_index).to_numpy()
        # Dampen heavy signals so a single power-user doesn't dominate a factor.
        vals = np.log1p(df["weight"].to_numpy())
        matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids), len(self.item_ids))
        )

        k = min(settings.CF_FACTORS, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=max(2, k), random_state=42)
        self.user_factors = svd.fit_transform(matrix)          # n_users x k
        self.item_factors = svd.components_.T                  # n_items x k
        return self

    def knows_user(self, user_id: str) -> bool:
        return user_id in self.user_index

    def scores_for_user(self, user_id: str) -> dict[str, float]:
        idx = self.user_index.get(user_id)
        if idx is None or self.user_factors is None or self.item_factors is None:
            return {}
        scores = self.item_factors @ self.user_factors[idx]
        return {self.item_ids[i]: float(scores[i]) for i in range(len(self.item_ids))}

    def recommend(self, user_id: str, k: int, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        idx = self.user_index.get(user_id)
        if idx is None or self.user_factors is None or self.item_factors is None:
            return []
        scores = self.item_factors @ self.user_factors[idx]
        exclude = exclude or set()
        ranked = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in ranked:
            cid = self.item_ids[i]
            if cid in exclude:
                continue
            out.append((cid, float(scores[i])))
            if len(out) >= k:
                break
        return out
