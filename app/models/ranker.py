"""Learning-to-rank layer: a LightGBM model that rescoring blended candidates
using richer features (CF score, content score, popularity, rating,
difficulty, duration). Trained on real outcomes (interacted = positive).

If the ranker artifact is absent, the recommender falls back to the weighted
ensemble blend — so this layer is strictly additive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

FEATURES = [
    "cf_score",
    "content_score",
    "pop",
    "rating_avg",
    "rating_count",
    "duration",
    "difficulty",
]

_DIFF = {"beginner": 0, "intermediate": 1, "advanced": 2}


def course_feature_table(courses: pd.DataFrame) -> pd.DataFrame:
    """Static per-course feature table, indexed by course_id."""
    feat = pd.DataFrame(index=courses["course_id"])
    feat["pop"] = np.log1p(courses["enrollment_count"].to_numpy())
    feat["rating_avg"] = courses["rating_avg"].to_numpy()
    feat["rating_count"] = np.log1p(courses["rating_count"].to_numpy())
    feat["duration"] = courses["duration_minutes"].to_numpy()
    feat["difficulty"] = [_DIFF.get(d, 0) for d in courses["difficulty_level"]]
    return feat


def feature_frame(
    cand_ids: list[str],
    cf_scores: dict[str, float],
    content_scores: dict[str, float],
    course_feat: pd.DataFrame,
) -> pd.DataFrame:
    base = course_feat.reindex(cand_ids)
    base.insert(0, "content_score", [content_scores.get(c, 0.0) for c in cand_ids])
    base.insert(0, "cf_score", [cf_scores.get(c, 0.0) for c in cand_ids])
    return base[FEATURES].fillna(0.0)


class RankerModel:
    def __init__(self) -> None:
        self.model: LGBMClassifier | None = None

    def train(self, X: pd.DataFrame, y: np.ndarray) -> "RankerModel":
        self.model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        self.model.fit(X[FEATURES], y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict_proba(X[FEATURES])[:, 1]


def build_training_set(
    interactions: pd.DataFrame,
    cf,
    content_profiles: dict[str, dict[str, float]],
    course_feat: pd.DataFrame,
    all_items: list[str],
    n_neg_per_pos: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """For each user: interacted items are positives; a random sample of
    non-interacted items are negatives. Features come from CF + content + course
    stats (identical to what serving computes)."""
    rng = np.random.default_rng(seed)
    by_user = interactions.groupby("user_id")["course_id"].apply(set).to_dict()
    item_arr = np.array(all_items)

    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []

    for user_id, pos_items in by_user.items():
        pos = [c for c in pos_items if c in course_feat.index]
        if not pos:
            continue
        n_neg = min(len(pos) * n_neg_per_pos, len(all_items) - len(pos))
        negs: list[str] = []
        if n_neg > 0:
            # Oversample so positives can be filtered out and still leave n_neg,
            # but never ask for more distinct items than the catalogue holds:
            # a learner with many positives relative to the catalogue would
            # otherwise push the request past the population size.
            pool_size = min(len(item_arr), n_neg * 2 + 10)
            pool = rng.choice(item_arr, size=pool_size, replace=False)
            negs = [c for c in pool if c not in pos_items][:n_neg]

        cand = pos + negs
        cf_scores = cf.scores_for_user(user_id)
        content_scores = content_profiles.get(user_id, {})
        X = feature_frame(cand, cf_scores, content_scores, course_feat)
        y = np.array([1] * len(pos) + [0] * len(negs))
        frames.append(X)
        labels.append(y)

    if not frames:
        return pd.DataFrame(columns=FEATURES), np.array([])
    return pd.concat(frames, ignore_index=True), np.concatenate(labels)
