"""Offline evaluation (RQ1): does the personalized hybrid beat the single
methods it is built from, and is the difference real rather than noise?

Protocol
--------
*Temporal* leave-last-N split. For every learner the interactions are ordered by
timestamp and the most recent `holdout_frac` are held out; models only ever see
the earlier ones. A random split would let a model train on a learner's future
and predict their past, which inflates every metric.

Blend weights are selected on a *validation* fold carved out of the training
history (the block just before the test block) and then frozen — the test fold is
scored once. Tuning on the test fold, as an earlier version of this pipeline did,
reports the best of many attempts rather than an honest held-out estimate.

Uncertainty comes from bootstrapping over test learners (the split itself is
deterministic), and the headline hybrid-vs-best-single comparison is checked with
a paired Wilcoxon signed-rank test over per-learner NDCG.

Beyond accuracy, the report includes catalogue coverage, novelty, intra-list
diversity and long-tail share, plus a cold/warm breakdown — a recommender that
wins on Recall by re-serving the same popular head is not a good recommender.

Known limitation: per-course `rating_avg` / `rating_count` features are computed
over the whole dataset, so a small amount of information from the test block
reaches the ranker's static features. Popularity, the dominant feature, is
recomputed from the training block only.

Run:  python -m app.pipelines.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from app.core.config import settings
from app.data.loaders import (
    load_course_text,
    load_courses,
    load_interactions,
    load_interest_signals,
)
from app.models import ranker as ranker_mod
from app.models.collaborative import CFModel
from app.models.content_based import ContentModel

# Learners with fewer than this many training interactions are "cold".
# Mirrors the backend's WARM_THRESHOLD so the breakdown matches live behaviour.
COLD_THRESHOLD = 3

SINGLE_METHODS = ("Popularity", "Content-based", "Collaborative")
ALL_METHODS = (*SINGLE_METHODS, "Hybrid", "Hybrid+Ranker")


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def _ordered(g: pd.DataFrame) -> pd.DataFrame:
    """Chronological order, with a stable sort so equal timestamps keep their
    original order rather than being shuffled between runs."""
    if "ts" in g.columns:
        return g.sort_values("ts", kind="mergesort")
    return g


def temporal_split(
    inter: pd.DataFrame, holdout_frac: float = 0.3, min_history: int = 2
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Per learner, hold out the most recent `holdout_frac` of their history.

    Learners with fewer than `min_history` interactions contribute their rows to
    training but are not scored — there is nothing to predict for them.
    """
    train_rows: list[dict] = []
    test: dict[str, set[str]] = {}

    for user, g in inter.groupby("user_id"):
        g = _ordered(g)
        records = g.to_dict("records")
        if len(records) < min_history:
            train_rows.extend(records)
            continue
        n_test = max(1, int(round(len(records) * holdout_frac)))
        n_test = min(n_test, len(records) - 1)  # always leave some history
        cut = len(records) - n_test
        train_rows.extend(records[:cut])
        held = {r["course_id"] for r in records[cut:]}
        if held:
            test[user] = held

    return pd.DataFrame(train_rows), test


def temporal_three_way(
    inter: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.3
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    """Chronological train / validation / test blocks per learner.

    The test block is the most recent slice; validation is the slice immediately
    before it. Hyperparameters are chosen against validation, so the test block
    stays untouched until the final scoring pass.
    """
    train_rows: list[dict] = []
    val: dict[str, set[str]] = {}
    test: dict[str, set[str]] = {}

    for user, g in inter.groupby("user_id"):
        g = _ordered(g)
        records = g.to_dict("records")
        n = len(records)
        if n < 3:  # too short to give up two blocks
            train_rows.extend(records)
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_test + n_val >= n:  # keep at least one training interaction
            n_val = max(0, n - n_test - 1)
        cut_val = n - n_test - n_val
        train_rows.extend(records[:cut_val])
        if n_val:
            val[user] = {r["course_id"] for r in records[cut_val : cut_val + n_val]}
        test[user] = {r["course_id"] for r in records[n - n_test :]}

    return pd.DataFrame(train_rows), val, test


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
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


class BeyondAccuracy:
    """Catalogue-level context for a ranking: novelty, diversity, long-tail share.

    Built once from the *training* block so nothing here peeks at held-out data.
    """

    def __init__(self, train: pd.DataFrame, content: ContentModel, all_items: list[str]):
        counts = train.groupby("course_id")["user_id"].nunique()
        n_users = max(1, train["user_id"].nunique())
        # Self-information: rarely-engaged courses carry more novelty.
        self.novelty = {
            c: float(-np.log2((counts.get(c, 0) + 1) / (n_users + 1))) for c in all_items
        }
        # "Head" = the 20% most-engaged courses; everything else is long tail.
        ranked = counts.sort_values(ascending=False)
        head_n = max(1, int(0.2 * len(all_items)))
        self.head = set(ranked.index[:head_n])
        self.content = content

    def novelty_of(self, rec: list[str]) -> float:
        vals = [self.novelty.get(c, 0.0) for c in rec]
        return float(np.mean(vals)) if vals else 0.0

    def long_tail_share(self, rec: list[str]) -> float:
        if not rec:
            return 0.0
        return float(sum(1 for c in rec if c not in self.head) / len(rec))

    def intra_list_diversity(self, rec: list[str]) -> float:
        """1 - mean pairwise cosine similarity between the recommended courses.
        A list of near-duplicates scores near 0; a varied list scores near 1."""
        idx = [self.content.id_to_idx[c] for c in rec if c in self.content.id_to_idx]
        if len(idx) < 2 or self.content.matrix is None:
            return 0.0
        sub = self.content.matrix[idx]           # rows are L2-normalized
        sims = (sub @ sub.T).toarray()
        n = len(idx)
        off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
        return float(1.0 - off)


def per_user_metrics(
    rec_lists: dict[str, list[str]],
    test: dict[str, set[str]],
    k: int,
    beyond: BeyondAccuracy,
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Score every test learner individually.

    Per-learner vectors (rather than only their means) are what make bootstrap
    intervals and the paired significance test possible.
    """
    out: dict[str, dict[str, float]] = {}
    covered: set[str] = set()
    for user, held in test.items():
        rec = rec_lists.get(user, [])[:k]
        covered.update(rec)
        hits = sum(1 for c in rec if c in held)
        out[user] = {
            "Recall": hits / len(held) if held else 0.0,
            "Precision": hits / k,
            "NDCG": _ndcg(rec, held, k),
            "MAP": _average_precision(rec, held, k),
            "MRR": _reciprocal_rank(rec, held, k),
            "HitRate": 1.0 if hits > 0 else 0.0,
            "Novelty": beyond.novelty_of(rec),
            "Diversity": beyond.intra_list_diversity(rec),
            "LongTail": beyond.long_tail_share(rec),
        }
    return out, covered


def _bootstrap_ci(
    values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean, resampling learners."""
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def aggregate(
    per_user: dict[str, dict[str, float]],
    covered: set[str],
    k: int,
    n_items: int,
    n_boot: int = 0,
    seed: int = 42,
) -> dict:
    """Mean of each per-learner metric, with the @K suffix the UI expects."""
    if not per_user:
        return {"Coverage": 0.0}
    names = next(iter(per_user.values())).keys()
    result: dict[str, float | dict] = {}
    for name in names:
        vals = np.array([m[name] for m in per_user.values()], dtype=float)
        result[f"{name}@{k}"] = float(vals.mean())
        if n_boot:
            # Kept at full precision: rounding the bounds can push them inside
            # the mean they are meant to bracket when a metric has no variance.
            lo, hi = _bootstrap_ci(vals, n_boot, seed)
            result[f"{name}@{k}_ci"] = {"lo": lo, "hi": hi}
    result["Coverage"] = len(covered) / n_items if n_items else 0.0
    return result


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test on per-learner scores (a vs b).

    Non-parametric because per-learner NDCG is bounded, zero-inflated and
    nowhere near normal, so a t-test would be the wrong instrument.
    """
    diff = a - b
    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        return {
            "test": "wilcoxon",
            "p_value": 1.0,
            "significant": False,
            "n_pairs": int(len(a)),
            "n_differing": 0,
            "median_diff": 0.0,
            "note": "identical rankings for every learner",
        }
    try:
        statistic, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:  # pragma: no cover — degenerate input
        return {"test": "wilcoxon", "p_value": 1.0, "significant": False, "note": str(exc)}
    # Rank-biserial correlation: a simple, bounded effect size for this test.
    n = n_nonzero
    total_rank = n * (n + 1) / 2
    effect = float(1 - 2 * statistic / total_rank) if total_rank else 0.0
    return {
        "test": "wilcoxon",
        "statistic": float(statistic),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "n_pairs": int(len(a)),
        "n_differing": n_nonzero,
        "median_diff": float(np.median(diff)),
        "effect_size_rank_biserial": round(effect, 3),
    }


# --------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------
def _minmax_over(d: dict[str, float], cand: list[str]) -> dict[str, float]:
    if not d:
        return {}
    v = np.array([d.get(c, 0.0) for c in cand], dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return {c: 0.0 for c in cand}
    return {c: (d.get(c, 0.0) - lo) / (hi - lo) for c in cand}


def _train_popularity_features(courses: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Course feature table whose popularity comes from the training block only.

    `courses.enrollment_count` is the live global count and therefore includes
    the held-out interactions; using it directly would leak the test block into
    the ranker's strongest feature.
    """
    counts = train.groupby("course_id")["user_id"].nunique()
    adjusted = courses.copy()
    adjusted["enrollment_count"] = (
        adjusted["course_id"].map(counts).fillna(0).astype(float)
    )
    return ranker_mod.course_feature_table(adjusted)


class _Fitted:
    """Models fitted on one training block, plus everything scoring needs."""

    def __init__(self, courses, text, train, all_items, seed: int = 42):
        merged = courses.merge(text, on="course_id", how="left")
        merged["text"] = merged["text"].fillna("")
        self.content = ContentModel().fit(
            merged["course_id"].tolist(), merged["text"].tolist()
        )
        self.cf = CFModel().fit(train, set(all_items))
        self.profiles = {
            u: dict(zip(g["course_id"], g["weight"])) for u, g in train.groupby("user_id")
        }
        pop = train.groupby("course_id")["weight"].sum().sort_values(ascending=False)
        self.pop_list = pop.index.tolist()
        self.pop_scores = {c: float(v) for c, v in pop.items()}
        self.all_items = all_items
        self.beyond = BeyondAccuracy(train, self.content, all_items)
        self.course_feat = _train_popularity_features(courses, train)

        # The served path re-scores a candidate union with LightGBM; train it on
        # the same training block so the evaluated model is the served model.
        self.ranker = None
        content_profiles = {
            u: self.content.profile_scores(items) for u, items in self.profiles.items()
        }
        X, y = ranker_mod.build_training_set(
            train, self.cf, content_profiles, self.course_feat, all_items, seed=seed
        )
        if len(y) and y.sum() > 0 and (y == 0).sum() > 0:
            self.ranker = ranker_mod.RankerModel().train(X, y)

    def scores_for(self, user: str) -> tuple[list[str], dict, dict, dict]:
        """Candidate list plus the three normalized signal scores for a learner."""
        profile = self.profiles.get(user, {})
        seen = set(profile)
        cand = [c for c in self.all_items if c not in seen]
        cf_raw = self.cf.scores_for_user(user)
        ct_raw = self.content.profile_scores(profile) if profile else {}
        return (
            cand,
            _minmax_over(cf_raw, cand),
            _minmax_over(ct_raw, cand),
            _minmax_over(self.pop_scores, cand),
        )

    def ranker_list(self, user: str, k: int) -> list[str]:
        """Reproduce serving: candidate generation from three sources, then
        LightGBM re-scoring over the union."""
        profile = self.profiles.get(user, {})
        seen = set(profile)
        n = settings.CAND_PER_SOURCE
        cf_ids = [c for c, _ in self.cf.recommend(user, n, exclude=seen)]
        ct_raw = self.content.profile_scores(profile) if profile else {}
        content_ids = [
            c for c, _ in sorted(ct_raw.items(), key=lambda x: -x[1]) if c not in seen
        ][:n]
        pop_ids = [c for c in self.pop_list if c not in seen][:n]
        cand = list(dict.fromkeys(cf_ids + content_ids + pop_ids))
        if not cand:
            return []
        if self.ranker is None:
            return cand[:k]
        X = ranker_mod.feature_frame(
            cand, self.cf.scores_for_user(user), ct_raw, self.course_feat
        )
        probs = self.ranker.score(X)
        order = np.argsort(-probs)
        return [cand[i] for i in order[:k]]


def _served_method(fit: _Fitted) -> str:
    """Which ablation arm corresponds to the configuration serving live traffic.

    Keeps the reported "served" row honest when `USE_RANKER` is toggled, so the
    evaluation never quietly describes a model the app does not run.
    """
    if settings.USE_RANKER and fit.ranker is not None:
        return "Hybrid+Ranker"
    return "Hybrid"


def _method_lists(
    fit: _Fitted, users, k: int, weights: tuple[float, float, float]
) -> dict[str, dict[str, list[str]]]:
    """Top-k list per method per learner, all from the same fitted models."""
    w_cf, w_ct, w_pop = weights
    lists: dict[str, dict[str, list[str]]] = {m: {} for m in ALL_METHODS}
    for user in users:
        cand, cfn, ctn, popn = fit.scores_for(user)
        seen = set(fit.profiles.get(user, {}))
        lists["Popularity"][user] = [c for c in fit.pop_list if c not in seen][:k]
        lists["Content-based"][user] = sorted(cand, key=lambda c: -ctn.get(c, 0.0))[:k]
        lists["Collaborative"][user] = sorted(cand, key=lambda c: -cfn.get(c, 0.0))[:k]
        blended = {
            c: w_cf * cfn.get(c, 0.0) + w_ct * ctn.get(c, 0.0) + w_pop * popn.get(c, 0.0)
            for c in cand
        }
        lists["Hybrid"][user] = sorted(cand, key=lambda c: -blended[c])[:k]
        lists["Hybrid+Ranker"][user] = fit.ranker_list(user, k)
    return lists


def _load_frames(include_interests: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Courses, per-course text, and the interaction matrix used for evaluation.

    Declared interests are folded in exactly as `train_pipeline` does, so the
    evaluated model sees the same signals as the deployed one. They carry no
    timestamp and are profile state rather than events, so they are attached to
    the earliest point in each learner's history — never to the held-out block.
    """
    courses = load_courses()
    text = load_course_text()
    inter = load_interactions()
    valid = set(courses["course_id"])
    inter = inter[inter["course_id"].isin(valid)].copy()
    inter["ts"] = pd.to_datetime(inter["ts"], utc=True, errors="coerce")
    # Marks rows that came from a real action (enrol / review / wishlist). Used
    # to segment cold vs warm learners: a declared interest expands into many
    # weak rows, which would otherwise make every learner look experienced.
    inter["is_event"] = 1.0

    if include_interests:
        signals = load_interest_signals()
        signals = signals[signals["course_id"].isin(valid)].copy()
        if len(signals):
            earliest = inter["ts"].min()
            if pd.isna(earliest):
                earliest = pd.Timestamp("1970-01-01", tz="UTC")
            # One step earlier than any real event, so a temporal split always
            # places declared interests in the training block.
            signals["ts"] = earliest - pd.Timedelta(seconds=1)
            signals["is_event"] = 0.0
            inter = pd.concat([inter, signals], ignore_index=True)
            inter = (
                inter.groupby(["user_id", "course_id"], as_index=False)
                .agg(weight=("weight", "sum"), ts=("ts", "max"),
                     is_event=("is_event", "max"))
            )
    return courses, text, inter


# --------------------------------------------------------------------------
# Blend tuning (on validation only)
# --------------------------------------------------------------------------
def tune_blend(k: int = 10, step: float = 0.05, val_frac: float = 0.15,
               test_frac: float = 0.3) -> dict:
    """Grid-search the blend weights against the VALIDATION block.

    Returns the winning weights and the validation NDCG that chose them. The
    test block is never consulted here, which is what makes the numbers reported
    by `run_ablation` an honest held-out estimate.
    """
    courses, text, inter = _load_frames()
    train, val, _ = temporal_three_way(inter, val_frac=val_frac, test_frac=test_frac)
    all_items = courses["course_id"].tolist()
    fit = _Fitted(courses, text, train, all_items)
    scored_users = [u for u in val if u in fit.profiles or fit.cf.knows_user(u)]

    grid: list[tuple[float, float, float]] = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            w_cf, w_ct = i * step, j * step
            grid.append((round(w_cf, 3), round(w_ct, 3), round(1.0 - w_cf - w_ct, 3)))

    # Score every learner's signals once; only the weighting changes per grid point.
    cached = {}
    for user in scored_users:
        cand, cfn, ctn, popn = fit.scores_for(user)
        cached[user] = (cand, cfn, ctn, popn)

    best, best_ndcg = None, -1.0
    results = []
    for w_cf, w_ct, w_pop in grid:
        ndcgs = []
        for user in scored_users:
            cand, cfn, ctn, popn = cached[user]
            blended = {
                c: w_cf * cfn.get(c, 0.0) + w_ct * ctn.get(c, 0.0) + w_pop * popn.get(c, 0.0)
                for c in cand
            }
            rec = sorted(cand, key=lambda c: -blended[c])[:k]
            ndcgs.append(_ndcg(rec, val[user], k))
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
        results.append({"cf": w_cf, "content": w_ct, "popularity": w_pop,
                        "val_ndcg": round(mean_ndcg, 4)})
        if mean_ndcg > best_ndcg:
            best, best_ndcg = (w_cf, w_ct, w_pop), mean_ndcg

    results.sort(key=lambda r: -r["val_ndcg"])
    return {
        "k": k,
        "best": {"cf": best[0], "content": best[1], "popularity": best[2]},
        "val_ndcg": round(best_ndcg, 4),
        "val_users": len(scored_users),
        "current": {
            "cf": settings.BLEND_CF,
            "content": settings.BLEND_CONTENT,
            "popularity": settings.BLEND_POPULARITY,
        },
        "top_10": results[:10],
        "grid_points": len(grid),
    }


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def run(k: int = 10, test_frac: float = 0.3, n_boot: int = 500) -> dict:
    """Headline RQ1 result: the served hybrid vs a popularity baseline."""
    courses, text, inter = _load_frames()
    train, test = temporal_split(inter, test_frac)
    all_items = courses["course_id"].tolist()
    n_items = len(all_items)
    print(f"Eval: {len(test)} test users, {len(train)} train interactions, {n_items} items")

    fit = _Fitted(courses, text, train, all_items)
    weights = (settings.BLEND_CF, settings.BLEND_CONTENT, settings.BLEND_POPULARITY)
    lists = _method_lists(fit, test.keys(), k, weights)

    hybrid_key = _served_method(fit)
    hu, hcov = per_user_metrics(lists[hybrid_key], test, k, fit.beyond)
    bu, bcov = per_user_metrics(lists["Popularity"], test, k, fit.beyond)
    hybrid = aggregate(hu, hcov, k, n_items, n_boot=n_boot)
    baseline = aggregate(bu, bcov, k, n_items, n_boot=n_boot)

    ndcg_h = np.array([hu[u]["NDCG"] for u in test])
    ndcg_b = np.array([bu[u]["NDCG"] for u in test])

    print("\n=== Offline evaluation (RQ1) ===")
    print(f"{'Metric':<16}{'Popularity':>14}{'Hybrid':>14}{'Lift':>10}")
    for key in hybrid:
        if key.endswith("_ci"):
            continue
        b, h = baseline.get(key, 0.0), hybrid[key]
        lift = f"{(h / b - 1) * 100:+.0f}%" if b else "n/a"
        print(f"{key:<16}{b:>14.4f}{h:>14.4f}{lift:>10}")

    return {
        "k": k,
        "protocol": "temporal_leave_last_n",
        "served_variant": hybrid_key,
        "test_users": len(test),
        "train_interactions": int(len(train)),
        "n_items": int(n_items),
        "hybrid": hybrid,
        "baseline": baseline,
        "significance": paired_test(ndcg_h, ndcg_b),
        "bootstrap_samples": n_boot,
    }


def run_ablation(
    k: int = 10, test_frac: float = 0.3, val_frac: float = 0.15, n_boot: int = 500
) -> dict:
    """RQ1 ablation: every method scored on the same temporal split.

    Popularity / Content-based / Collaborative / Hybrid (blend) / Hybrid+Ranker
    (the configuration actually served). Blend weights come from the validation
    block, so the test numbers are not the product of tuning against them.
    """
    courses, text, inter = _load_frames()
    train, val, test = temporal_three_way(inter, val_frac=val_frac, test_frac=test_frac)
    all_items = courses["course_id"].tolist()
    n_items = len(all_items)

    fit = _Fitted(courses, text, train, all_items)
    weights = (settings.BLEND_CF, settings.BLEND_CONTENT, settings.BLEND_POPULARITY)
    lists = _method_lists(fit, test.keys(), k, weights)

    per_user: dict[str, dict[str, dict[str, float]]] = {}
    methods: dict[str, dict] = {}
    for name in ALL_METHODS:
        pu, cov = per_user_metrics(lists[name], test, k, fit.beyond)
        per_user[name] = pu
        methods[name] = aggregate(pu, cov, k, n_items, n_boot=n_boot)

    # Cold vs warm: does the hybrid actually help learners with little history?
    # Counted over real actions only — declared interests expand into many weak
    # rows, so including them would classify every learner as warm.
    if "is_event" in train.columns:
        real_history = train[train["is_event"] > 0].groupby("user_id").size().to_dict()
    else:
        real_history = train.groupby("user_id").size().to_dict()
    cold_users = [u for u in test if real_history.get(u, 0) < COLD_THRESHOLD]
    cold_set = set(cold_users)
    warm_users = [u for u in test if u not in cold_set]
    segments: dict[str, dict] = {}
    for seg_name, seg_users in (("cold", cold_users), ("warm", warm_users)):
        if not seg_users:
            continue
        segments[seg_name] = {
            "n_users": len(seg_users),
            "methods": {
                name: aggregate(
                    {u: per_user[name][u] for u in seg_users},
                    set().union(*(set(lists[name].get(u, [])[:k]) for u in seg_users)),
                    k,
                    n_items,
                )
                for name in ALL_METHODS
            },
        }

    # Is the hybrid's edge over the best single method real?
    ndcg_key = f"NDCG@{k}"
    best_single = max(SINGLE_METHODS, key=lambda m: methods[m][ndcg_key])
    best_overall = max(ALL_METHODS, key=lambda m: methods[m][ndcg_key])
    served = _served_method(fit)

    def _ndcg_vec(name: str) -> np.ndarray:
        return np.array([per_user[name][u]["NDCG"] for u in test])

    def _compare(a: str, b: str) -> dict:
        return {"comparison": f"{a} vs {b}", "a": a, "b": b,
                "mean_a": round(methods[a][ndcg_key], 4),
                "mean_b": round(methods[b][ndcg_key], 4),
                **paired_test(_ndcg_vec(a), _ndcg_vec(b))}

    # The headline RQ1 claim, plus the two comparisons that keep it honest:
    # whether the deployed configuration is the one that actually won, and
    # whether the learned ranker earns its place over the plain blend.
    comparisons = [_compare(best_overall, best_single)]
    if served != best_overall:
        comparisons.append(_compare(served, best_single))
    if "Hybrid" in ALL_METHODS and "Hybrid+Ranker" in ALL_METHODS:
        comparisons.append(_compare("Hybrid", "Hybrid+Ranker"))

    significance = {
        **comparisons[0],  # backwards-compatible flat shape
        "best_single_method": best_single,
        "best_overall_method": best_overall,
        "served_method": served,
        "comparisons": comparisons,
    }

    return {
        "k": k,
        "protocol": "temporal_leave_last_n",
        "served_variant": served,
        "test_users": len(test),
        "val_users": len(val),
        "train_interactions": int(len(train)),
        "n_items": int(n_items),
        "blend": {
            "cf": settings.BLEND_CF,
            "content": settings.BLEND_CONTENT,
            "popularity": settings.BLEND_POPULARITY,
            "selected_on": "validation_block",
        },
        "methods": methods,
        "segments": segments,
        "significance": significance,
        "bootstrap_samples": n_boot,
    }


def _print_ablation(res: dict, k: int) -> None:
    print(f"\n=== RQ1 method ablation ({res['protocol']}, K={k}) ===")
    print(f"{res['test_users']} test learners · {res['n_items']} items · "
          f"blend cf {res['blend']['cf']} / content {res['blend']['content']} / "
          f"pop {res['blend']['popularity']} (selected on {res['blend']['selected_on']})")
    header = f"{'Method':<16}{'NDCG':>9}{'95% CI':>18}{'Recall':>9}{'HitRate':>9}{'Novelty':>9}{'LongTail':>10}{'Coverage':>10}"
    print(header)
    for name, m in res["methods"].items():
        ci = m.get(f"NDCG@{k}_ci", {})
        ci_s = f"[{ci.get('lo', 0):.3f}, {ci.get('hi', 0):.3f}]"
        print(f"{name:<16}{m[f'NDCG@{k}']:>9.4f}{ci_s:>18}{m[f'Recall@{k}']:>9.4f}"
              f"{m[f'HitRate@{k}']:>9.4f}{m[f'Novelty@{k}']:>9.3f}"
              f"{m[f'LongTail@{k}']:>10.3f}{m['Coverage']:>10.4f}")
    print()
    for cmp in res["significance"]["comparisons"]:
        verdict = "significant" if cmp["significant"] else "n.s."
        print(f"  {cmp['comparison']:<34} {cmp['mean_a']:.4f} vs {cmp['mean_b']:.4f}  "
              f"p = {cmp['p_value']:.2e} ({verdict}), effect "
              f"{cmp.get('effect_size_rank_biserial', 0)}")
    for seg, d in res.get("segments", {}).items():
        best = max(d["methods"], key=lambda m: d["methods"][m][f"NDCG@{k}"])
        print(f"  {seg:<5} learners (n={d['n_users']:>3}): best = {best} "
              f"@ NDCG {d['methods'][best][f'NDCG@{k}']:.4f}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "tune":
        res = tune_blend()
        print("Best blend on the validation block:", res["best"],
              f"(val NDCG@{res['k']} = {res['val_ndcg']})")
        print("Currently configured:", res["current"])
        print("\nTop candidates:")
        for row in res["top_10"]:
            print(f"  cf={row['cf']:.2f} content={row['content']:.2f} "
                  f"pop={row['popularity']:.2f}  val_ndcg={row['val_ndcg']:.4f}")
    elif mode == "ablation":
        _print_ablation(run_ablation(), 10)
    else:
        run()
