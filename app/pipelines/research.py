"""Research analyses that turn the platform's data into evidence for the two
thesis research questions.

RQ1 — does a hybrid (content + collaborative) recommender beat the single methods
       it is built from?  -> `evaluate.run_ablation` (Popularity / Content / CF / Hybrid).

RQ2 — can anonymized interaction signals (view frequency, time-on-lesson,
       completion) act as proxies for learning preference and difficulty
       alignment?  -> `rq2()` below, which measures how recoverable those latent
       properties are from the raw signals alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.db import read_sql
from app.pipelines import evaluate as evaluate_mod

DIFFS = ["beginner", "intermediate", "advanced"]
_DI = {d: i for i, d in enumerate(DIFFS)}


# --- SQL --------------------------------------------------------------------
_PROGRESS_SQL = """
SELECT lp.user_id::text                      AS user_id,
       c.difficulty_level::text              AS difficulty,
       s.id::text                            AS sector_id,
       s.name                                AS sector_name,
       lp.watched_seconds                    AS watched,
       GREATEST(l.duration_seconds, 30)      AS duration
FROM lesson_progress lp
JOIN lessons l    ON l.id = lp.lesson_id
JOIN courses c    ON c.id = lp.course_id
JOIN categories f ON f.id = c.category_id
JOIN categories s ON s.id = f.parent_category_id
WHERE lp.watched_seconds > 0
"""

_INTERESTS_SQL = """
SELECT user_id::text AS user_id, category_id::text AS sector_id, weight
FROM user_interests
"""

_VIEWS_SQL = """
SELECT object_id::text AS course_id, count(*) AS views
FROM interaction_events
WHERE event_type = 'view_course' AND object_id IS NOT NULL
GROUP BY object_id
"""

_ENROLL_SQL = """
SELECT course_id::text AS course_id, count(*) AS enrollments
FROM enrollments GROUP BY course_id
"""


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _difficulty_alignment(prog: pd.DataFrame) -> dict:
    """Time-on-lesson as a proxy for difficulty fit.

    For every learner we measure engagement (watched / lesson length) per course
    difficulty. Their 'revealed level' is the difficulty they invest most in.
    Two checks:
      * reliability: estimate the revealed level from a random half of a learner's
        lessons and again from the other half — how often do the halves agree?
        A real proxy must be reproducible, not noise (chance = 1/3).
      * structure: the average engagement grouped by (revealed level x difficulty)
        should peak on the diagonal.
    """
    prog = prog.copy()
    prog["ratio"] = prog["watched"] / prog["duration"]
    rng = np.random.default_rng(42)

    revealed: dict[str, int] = {}
    agree = 0
    reliab_n = 0
    # accumulate engagement matrix rows=revealed level, cols=difficulty
    mat_sum = np.zeros((3, 3))
    mat_cnt = np.zeros((3, 3))

    for uid, g in prog.groupby("user_id"):
        by_diff = g.groupby("difficulty")["ratio"].mean()
        # revealed level = difficulty with the highest relative engagement
        best = max(by_diff.items(), key=lambda kv: kv[1])[0]
        lvl = _DI[best]
        revealed[uid] = lvl
        for d, r in by_diff.items():
            mat_sum[lvl, _DI[d]] += r
            mat_cnt[lvl, _DI[d]] += 1

        # split-half reliability (needs a few lessons on each side)
        if len(g) >= 6:
            idx = rng.permutation(len(g))
            half = len(idx) // 2
            a = g.iloc[idx[:half]]
            b = g.iloc[idx[half:]]
            la = a.groupby("difficulty")["ratio"].mean()
            lb = b.groupby("difficulty")["ratio"].mean()
            if len(la) and len(lb):
                reliab_n += 1
                if la.idxmax() == lb.idxmax():
                    agree += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(mat_cnt > 0, mat_sum / mat_cnt, 0.0)

    dist = {DIFFS[i]: int(sum(1 for v in revealed.values() if v == i)) for i in range(3)}
    return {
        "matrix": [[round(float(x), 3) for x in row] for row in matrix],
        "matrix_counts": [[int(x) for x in row] for row in mat_cnt],
        "levels": DIFFS,
        "reliability": round(agree / reliab_n, 3) if reliab_n else 0.0,
        "reliability_hits": int(agree),
        "chance": round(1 / 3, 3),
        "n_learners": int(prog["user_id"].nunique()),
        "n_reliability_learners": reliab_n,
        "revealed_distribution": dist,
    }


def _preference_proxy(prog: pd.DataFrame, interests: pd.DataFrame) -> dict:
    """Does the sector a learner spends the most time in match the sector they
    actually prefer (their strongest declared/evolved interest)?"""
    eng = prog.groupby(["user_id", "sector_id"])["watched"].sum().reset_index()
    top_eng = eng.sort_values("watched", ascending=False).groupby("user_id").head(1)
    top_eng_map = dict(zip(top_eng["user_id"], top_eng["sector_id"]))

    top_int = interests.sort_values("weight", ascending=False).groupby("user_id").head(1)
    top_int_map = dict(zip(top_int["user_id"], top_int["sector_id"]))

    # top-3 engagement sectors for the softer overlap metric
    eng_sorted = eng.sort_values("watched", ascending=False)
    top3_eng = eng_sorted.groupby("user_id")["sector_id"].apply(lambda s: set(s.head(3)))
    int_sorted = interests.sort_values("weight", ascending=False)
    top3_int = int_sorted.groupby("user_id")["sector_id"].apply(lambda s: set(s.head(3)))

    shared = set(top_eng_map) & set(top_int_map)
    hits = [top_eng_map[u] == top_int_map[u] for u in shared]
    top1 = np.mean(hits) if hits else 0.0
    overlap = [
        len(top3_eng[u] & top3_int[u]) / max(1, len(top3_int[u]))
        for u in (set(top3_eng.index) & set(top3_int.index))
    ]
    n_sectors = interests["sector_id"].nunique()
    return {
        "top1_agreement": round(float(top1), 3),
        "top1_matches": int(sum(hits)),
        "top3_overlap": round(float(np.mean(overlap)) if overlap else 0.0, 3),
        "n_top3_learners": len(overlap),
        "chance": round(1 / n_sectors, 3) if n_sectors else 0.0,
        "n_sectors": int(n_sectors),
        "n_learners": len(shared),
    }


def _view_frequency() -> dict:
    """Course view frequency (anonymized clickstream) vs actual enrollment demand."""
    views = read_sql(_VIEWS_SQL)
    enr = read_sql(_ENROLL_SQL)
    df = views.merge(enr, on="course_id", how="inner")
    if df.empty:
        return {"pearson": 0.0, "spearman": 0.0, "n_courses": 0, "scatter": []}
    x = df["views"].to_numpy(float)
    y = df["enrollments"].to_numpy(float)
    pearson = _pearson(x, y)
    spearman = _pearson(
        df["views"].rank().to_numpy(float), df["enrollments"].rank().to_numpy(float)
    )
    sample = df.sample(min(250, len(df)), random_state=7)
    scatter = [{"views": int(v), "enrollments": int(e)} for v, e in zip(sample["views"], sample["enrollments"])]
    return {
        "pearson": round(pearson, 3),
        "spearman": round(spearman, 3),
        "n_courses": int(len(df)),
        "scatter": scatter,
    }


def rq1(k: int = 10) -> dict:
    return evaluate_mod.run_ablation(k=k)


def rq2() -> dict:
    prog = read_sql(_PROGRESS_SQL)
    interests = read_sql(_INTERESTS_SQL)
    if prog.empty:
        return {"available": False, "reason": "No time-on-lesson data. Run app.seeds.research_signals."}
    return {
        "available": True,
        "difficulty": _difficulty_alignment(prog),
        "preference": _preference_proxy(prog, interests),
        "view_frequency": _view_frequency(),
        "signals": {
            "n_progress_rows": int(len(prog)),
            "n_learners": int(prog["user_id"].nunique()),
        },
    }
