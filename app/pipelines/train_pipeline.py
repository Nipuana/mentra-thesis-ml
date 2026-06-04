"""Offline training: build content vectors, CF factors, the ranker, and persist
all artifacts. Run:  python -m app.pipelines.train_pipeline
"""

from __future__ import annotations

import pandas as pd

from app.data.loaders import (
    load_course_text,
    load_courses,
    load_interactions,
    load_interest_signals,
)
from app.models.collaborative import CFModel
from app.models.content_based import ContentModel
from app.models.ranker import RankerModel, build_training_set, course_feature_table
from app.registry import store


def run() -> dict:
    print("Loading data from Postgres…")
    courses = load_courses()
    text = load_course_text()
    interactions = load_interactions()

    merged = courses.merge(text, on="course_id", how="left")
    merged["text"] = merged["text"].fillna("")
    valid_items = set(courses["course_id"])
    inter = interactions[interactions["course_id"].isin(valid_items)].copy()

    # Fold declared/evolved interests in as weak signals, then re-aggregate so
    # each (user, course) has a single summed weight. This is how the AI "picks
    # up" onboarding keywords and interest drift.
    interest_signals = load_interest_signals()
    interest_signals = interest_signals[interest_signals["course_id"].isin(valid_items)]
    n_interest = len(interest_signals)
    if n_interest:
        inter = pd.concat([inter, interest_signals], ignore_index=True)
        inter = inter.groupby(["user_id", "course_id"], as_index=False)["weight"].sum()

    print(f"  {len(courses)} courses, {len(inter)} interactions "
          f"({n_interest} from interests), {inter['user_id'].nunique()} users")

    print("Fitting content model (TF-IDF over transcripts + metadata)…")
    content = ContentModel().fit(merged["course_id"].tolist(), merged["text"].tolist())

    print("Fitting collaborative filtering (SVD latent factors)…")
    cf = CFModel().fit(inter, valid_items)

    # user -> {course_id: weight}
    user_profiles = {
        u: dict(zip(g["course_id"], g["weight"])) for u, g in inter.groupby("user_id")
    }

    print("Building ranker training set + training LightGBM…")
    content_profiles = {u: content.profile_scores(items) for u, items in user_profiles.items()}
    course_feat = course_feature_table(courses)
    X, y = build_training_set(inter, cf, content_profiles, course_feat, courses["course_id"].tolist())

    ranker = None
    if len(y) and y.sum() > 0 and (y == 0).sum() > 0:
        ranker = RankerModel().train(X, y)

    print("Saving artifacts…")
    store.save("content", content)
    store.save("cf", cf)
    store.save("courses", courses)
    store.save("user_profiles", user_profiles)
    if ranker is not None:
        store.save("ranker", ranker)

    meta = {
        "courses": int(len(courses)),
        "interactions": int(len(inter)),
        "users": int(inter["user_id"].nunique()),
        "cf_factors": cf.user_factors.shape[1] if cf.user_factors is not None else 0,
        "ranker_rows": int(len(y)),
        "ranker_positives": int(y.sum()) if len(y) else 0,
        "has_ranker": ranker is not None,
    }
    store.write_manifest(meta)
    print("Done:", meta)
    return meta


if __name__ == "__main__":
    run()
