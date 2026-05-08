"""Content-based model: TF-IDF course vectors over transcripts + metadata.

Interface (vectorizer + normalized matrix) is a drop-in seam for swapping in
sentence-transformer embeddings later — only `fit` / `_vec` change.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from app.core.config import settings


class ContentModel:
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: sparse.csr_matrix | None = None  # L2-normalized rows
        self.course_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}

    def fit(self, course_ids: list[str], texts: list[str]) -> "ContentModel":
        self.vectorizer = TfidfVectorizer(
            max_features=settings.TFIDF_MAX_FEATURES,
            min_df=settings.TFIDF_MIN_DF,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        matrix = self.vectorizer.fit_transform(texts)
        self.matrix = normalize(matrix, norm="l2", axis=1).tocsr()
        self.course_ids = list(course_ids)
        self.id_to_idx = {cid: i for i, cid in enumerate(self.course_ids)}
        return self

    # --- similarity ---
    def similar(self, course_id: str, k: int = 10) -> list[tuple[str, float]]:
        idx = self.id_to_idx.get(course_id)
        if idx is None or self.matrix is None:
            return []
        sims = self.matrix.dot(self.matrix[idx].T).toarray().ravel()
        sims[idx] = -1.0  # exclude self
        return self._topk(sims, k)

    # --- user content profile (weighted average of engaged course vectors) ---
    def profile_scores(self, items_weights: dict[str, float]) -> dict[str, float]:
        if self.matrix is None:
            return {}
        rows, weights = [], []
        for cid, w in items_weights.items():
            idx = self.id_to_idx.get(cid)
            if idx is not None:
                rows.append(idx)
                weights.append(w)
        if not rows:
            return {}
        profile = self.matrix[rows].multiply(np.array(weights)[:, None]).sum(axis=0)
        profile = normalize(np.asarray(profile), norm="l2")  # 1 x V
        scores = self.matrix.dot(sparse.csr_matrix(profile).T).toarray().ravel()
        return {cid: float(scores[i]) for cid, i in self.id_to_idx.items()}

    # --- semantic-ish query (same vector space as courses) ---
    def query_scores(self, text: str) -> dict[str, float]:
        if self.vectorizer is None or self.matrix is None:
            return {}
        qv = normalize(self.vectorizer.transform([text]), norm="l2")
        scores = self.matrix.dot(qv.T).toarray().ravel()
        return {cid: float(scores[i]) for cid, i in self.id_to_idx.items()}

    def _topk(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.course_ids[i], float(scores[i])) for i in idx if scores[i] > 0]
