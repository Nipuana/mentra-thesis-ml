"""Offline evaluation (RQ1): does the personalized hybrid beat a popularity
baseline? Leave-out split per user, then Recall@K / Precision@K / NDCG@K /
coverage for the blended recommender vs recommending the globally-popular list.

Run:  python -m app.pipelines.evaluate
"""

from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.data.loaders import load_course_text, load_courses, load_interactions
from app.models.collaborative import CFModel
from app.models.content_based import ContentModel


def _split(inter, test_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    train_rows, test = [], {}
    for user, g in inter.groupby("user_id"):
        items = g["course_id"].tolist()
        if len(items) < 2:
            train_rows.extend(g.to_dict("records"))
            continue
        n_test = max(1, int(round(len(items) * test_frac)))
        idx = rng.permutation(len(items))
        test_idx = set(idx[:n_test].tolist())
        held = set()
        for i, rec in enumerate(g.to_dict("records")):
            if i in test_idx:
                held.add(rec["course_id"])
            else:
                train_rows.append(rec)
        if held:
            test[user] = held
    import pandas as pd

    return pd.DataFrame(train_rows), test


def _ndcg(recommended: list[str], held: set[str], k: int) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, c in enumerate(recommended[:k]) if c in held)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(held), k)))
    return dcg / ideal if ideal > 0 else 0.0


def _average_precision(rec: list[str], held: set[str], k: int) -> float:
    """Average Precision@K: precision measured at each rank where a relevant
    course appears, averaged. Rewards clustering the right courses near the top."""
    if not held:
        return 0.0
    hits = 0
    score = 0.0
    for i, c in enumerate(rec[:k]):
        if c in held:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(held), k)


def _reciprocal_rank(rec: list[str], held: set[str], k: int) -> float:
    """1 / rank of the first relevant course (0 if none in the top K)."""
    for i, c in enumerate(rec[:k]):
        if c in held:
            return 1.0 / (i + 1)
    return 0.0


def _metrics(rec_lists: dict[str, list[str]], test: dict[str, set[str]], k: int, n_items: int):
    recalls, precisions, ndcgs, aps, rrs, hitrates, covered = [], [], [], [], [], [], set()
    for user, held in test.items():
        rec = rec_lists.get(user, [])[:k]
        covered.update(rec)
        hits = sum(1 for c in rec if c in held)
        recalls.append(hits / len(held))
        precisions.append(hits / k)
        ndcgs.append(_ndcg(rec, held, k))
        aps.append(_average_precision(rec, held, k))
        rrs.append(_reciprocal_rank(rec, held, k))
        hitrates.append(1.0 if hits > 0 else 0.0)
    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"Precision@{k}": float(np.mean(precisions)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"MAP@{k}": float(np.mean(aps)),
        f"MRR@{k}": float(np.mean(rrs)),
        f"HitRate@{k}": float(np.mean(hitrates)),
        "Coverage": len(covered) / n_items,
    }


def run(k: int = 10, test_frac: float = 0.3) -> dict:
    courses = load_courses()
    text = load_course_text()
    inter_all = load_interactions()
    valid = set(courses["course_id"])
    inter = inter_all[inter_all["course_id"].isin(valid)].copy()

    train, test = _split(inter, test_frac)
    print(f"Eval: {len(test)} test users, {len(train)} train interactions, {len(courses)} items")

    merged = courses.merge(text, on="course_id", how="left")
    merged["text"] = merged["text"].fillna("")
    content = ContentModel().fit(merged["course_id"].tolist(), merged["text"].tolist())
    cf = CFModel().fit(train, valid)

    profiles = {u: dict(zip(g["course_id"], g["weight"])) for u, g in train.groupby("user_id")}

    # popularity from the training split
    pop = train.groupby("course_id")["weight"].sum().sort_values(ascending=False)
    pop_list = pop.index.tolist()
    pop_scores = {c: float(v) for c, v in pop.items()}
    all_items = courses["course_id"].tolist()

    hybrid_recs: dict[str, list[str]] = {}
    baseline_recs: dict[str, list[str]] = {}

    for user in test:
        profile = profiles.get(user, {})
        seen = set(profile)

        # --- popularity baseline ---
        baseline_recs[user] = [c for c in pop_list if c not in seen][:k]

        # --- personalized hybrid blend (cf + content + popularity) ---
        cf_scores = cf.scores_for_user(user)
        content_scores = content.profile_scores(profile) if profile else {}
        cand = [c for c in all_items if c not in seen]

        def mm(d):
            if not d:
                return {}
            v = np.array([d.get(c, 0.0) for c in cand], float)
            lo, hi = v.min(), v.max()
            return {c: (d.get(c, 0.0) - lo) / (hi - lo) if hi > lo else 0.0 for c in cand}

        cfn, ctn = mm(cf_scores), mm(content_scores)
        popn = mm(pop_scores)
        blended = {
            c: settings.BLEND_CF * cfn.get(c, 0.0)
            + settings.BLEND_CONTENT * ctn.get(c, 0.0)
            + settings.BLEND_POPULARITY * popn.get(c, 0.0)
            for c in cand
        }
        hybrid_recs[user] = sorted(cand, key=lambda c: -blended[c])[:k]

    n = len(courses)
    hybrid = _metrics(hybrid_recs, test, k, n)
    baseline = _metrics(baseline_recs, test, k, n)

    print("\n=== Offline evaluation (RQ1) ===")
    print(f"{'Metric':<14}{'Popularity':>14}{'Hybrid':>14}{'Lift':>10}")
    for key in hybrid:
        b, h = baseline[key], hybrid[key]
        lift = f"{(h / b - 1) * 100:+.0f}%" if b > 0 else "n/a"
        print(f"{key:<14}{b:>14.4f}{h:>14.4f}{lift:>10}")
    return {
        "k": k,
        "test_users": len(test),
        "train_interactions": int(len(train)),
        "n_items": int(n),
        "hybrid": hybrid,
        "baseline": baseline,
    }


def run_ablation(k: int = 10, test_frac: float = 0.3) -> dict:
    """RQ1 ablation: score four recommenders on the SAME leave-out split so the
    hybrid can be compared to each single method it combines.

    Methods:
      - Popularity   : non-personalized bestsellers (baseline)
      - Content-based: TF-IDF similarity to the learner's own history
      - Collaborative: SVD latent-factor CF over the learner-course matrix
      - Hybrid       : the production blend of all three
    """
    courses = load_courses()
    text = load_course_text()
    inter_all = load_interactions()
    valid = set(courses["course_id"])
    inter = inter_all[inter_all["course_id"].isin(valid)].copy()

    train, test = _split(inter, test_frac)

    merged = courses.merge(text, on="course_id", how="left")
    merged["text"] = merged["text"].fillna("")
    content = ContentModel().fit(merged["course_id"].tolist(), merged["text"].tolist())
    cf = CFModel().fit(train, valid)

    profiles = {u: dict(zip(g["course_id"], g["weight"])) for u, g in train.groupby("user_id")}
    pop = train.groupby("course_id")["weight"].sum().sort_values(ascending=False)
    pop_list = pop.index.tolist()
    pop_scores = {c: float(v) for c, v in pop.items()}
    all_items = courses["course_id"].tolist()
    n = len(courses)

    recs: dict[str, dict[str, list[str]]] = {m: {} for m in ("Popularity", "Content-based", "Collaborative", "Hybrid")}

    for user in test:
        profile = profiles.get(user, {})
        seen = set(profile)
        cand = [c for c in all_items if c not in seen]

        recs["Popularity"][user] = [c for c in pop_list if c not in seen][:k]

        cf_scores = cf.scores_for_user(user)
        content_scores = content.profile_scores(profile) if profile else {}

        def mm(d):
            if not d:
                return {}
            v = np.array([d.get(c, 0.0) for c in cand], float)
            lo, hi = v.min(), v.max()
            return {c: (d.get(c, 0.0) - lo) / (hi - lo) if hi > lo else 0.0 for c in cand}

        cfn, ctn, popn = mm(cf_scores), mm(content_scores), mm(pop_scores)

        recs["Content-based"][user] = sorted(cand, key=lambda c: -ctn.get(c, 0.0))[:k]
        recs["Collaborative"][user] = sorted(cand, key=lambda c: -cfn.get(c, 0.0))[:k]
        blended = {
            c: settings.BLEND_CF * cfn.get(c, 0.0)
            + settings.BLEND_CONTENT * ctn.get(c, 0.0)
            + settings.BLEND_POPULARITY * popn.get(c, 0.0)
            for c in cand
        }
        recs["Hybrid"][user] = sorted(cand, key=lambda c: -blended[c])[:k]

    methods = {m: _metrics(recs[m], test, k, n) for m in recs}
    return {
        "k": k,
        "test_users": len(test),
        "train_interactions": int(len(train)),
        "n_items": int(n),
        "blend": {
            "cf": settings.BLEND_CF,
            "content": settings.BLEND_CONTENT,
            "popularity": settings.BLEND_POPULARITY,
        },
        "methods": methods,
    }


if __name__ == "__main__":
    run()
