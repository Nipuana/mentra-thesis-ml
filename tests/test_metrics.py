"""Metric definitions, checked against hand-computed values.

Every headline number in the thesis is one of these functions applied to a
ranking, so a silent error here would be invisible and total.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.models.content_based import ContentModel
from app.pipelines.evaluate import (
    BeyondAccuracy,
    _average_precision,
    _bootstrap_ci,
    _ndcg,
    _reciprocal_rank,
    aggregate,
    paired_test,
    per_user_metrics,
)


# --- NDCG -------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    assert _ndcg(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_is_zero_without_hits():
    assert _ndcg(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_rewards_higher_placement():
    first = _ndcg(["a", "x", "y"], {"a"}, 3)
    last = _ndcg(["x", "y", "a"], {"a"}, 3)
    assert first > last


def test_ndcg_matches_hand_computation():
    # One relevant item at rank 2 -> DCG = 1/log2(3); ideal (single item) = 1.
    expected = 1 / math.log2(3)
    assert _ndcg(["x", "a", "y"], {"a"}, 3) == pytest.approx(expected)


def test_ndcg_ideal_is_capped_at_k():
    """With more relevant items than slots, a full list must still score 1.0."""
    assert _ndcg(["a", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(1.0)


# --- MAP / MRR --------------------------------------------------------------
def test_average_precision_hand_computation():
    # Hits at ranks 1 and 3 -> (1/1 + 2/3) / 2
    assert _average_precision(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_without_relevant_items():
    assert _average_precision(["a"], set(), 1) == 0.0


def test_reciprocal_rank_uses_first_hit():
    assert _reciprocal_rank(["x", "y", "a"], {"a"}, 3) == pytest.approx(1 / 3)
    assert _reciprocal_rank(["x"], {"a"}, 1) == 0.0


# --- Beyond-accuracy --------------------------------------------------------
@pytest.fixture
def beyond() -> BeyondAccuracy:
    # c0 is engaged by everyone, c4 by nobody -> a clear head and tail.
    train = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u1", "u2"],
            "course_id": ["c0", "c0", "c0", "c1", "c2"],
            "weight": 1.0,
        }
    )
    items = ["c0", "c1", "c2", "c3", "c4"]
    content = ContentModel().fit(
        items,
        [
            "python programming code",
            "python programming code",   # near-duplicate of c0
            "baroque violin sonata",
            "watercolour landscape painting",
            "quantum field theory",
        ],
    )
    return BeyondAccuracy(train, content, items)


def test_novelty_is_higher_for_rare_courses(beyond):
    assert beyond.novelty["c4"] > beyond.novelty["c0"]


def test_long_tail_share_excludes_the_popular_head(beyond):
    assert beyond.long_tail_share(["c0"]) == 0.0
    assert beyond.long_tail_share(["c3", "c4"]) == 1.0
    assert beyond.long_tail_share(["c0", "c4"]) == pytest.approx(0.5)


def test_diversity_penalises_near_duplicate_lists(beyond):
    duplicates = beyond.intra_list_diversity(["c0", "c1"])   # same text
    varied = beyond.intra_list_diversity(["c0", "c3"])       # unrelated text
    assert varied > duplicates


def test_diversity_of_a_single_item_list_is_zero(beyond):
    assert beyond.intra_list_diversity(["c0"]) == 0.0


# --- Aggregation ------------------------------------------------------------
def test_per_user_metrics_and_coverage(beyond):
    recs = {"u1": ["c0", "c1"], "u2": ["c2", "c3"]}
    held = {"u1": {"c1"}, "u2": {"zzz"}}
    per_user, covered = per_user_metrics(recs, held, k=2, beyond=beyond)

    assert per_user["u1"]["HitRate"] == 1.0
    assert per_user["u2"]["HitRate"] == 0.0
    assert per_user["u1"]["Recall"] == pytest.approx(1.0)
    assert covered == {"c0", "c1", "c2", "c3"}

    agg = aggregate(per_user, covered, k=2, n_items=5)
    assert agg["HitRate@2"] == pytest.approx(0.5)
    assert agg["Coverage"] == pytest.approx(4 / 5)


def test_aggregate_adds_intervals_only_when_bootstrapping(beyond):
    recs = {f"u{i}": ["c0", "c1"] for i in range(20)}
    held = {f"u{i}": {"c1"} for i in range(20)}
    per_user, covered = per_user_metrics(recs, held, k=2, beyond=beyond)

    plain = aggregate(per_user, covered, k=2, n_items=5)
    assert "NDCG@2_ci" not in plain

    with_ci = aggregate(per_user, covered, k=2, n_items=5, n_boot=100)
    ci = with_ci["NDCG@2_ci"]
    assert ci["lo"] <= with_ci["NDCG@2"] <= ci["hi"]


# --- Uncertainty and significance ------------------------------------------
def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=200)
    lo, hi = _bootstrap_ci(values, n_boot=500, seed=1)
    assert lo < values.mean() < hi


def test_bootstrap_interval_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = _bootstrap_ci(rng.normal(0.5, 0.1, 30), n_boot=500, seed=1)
    large = _bootstrap_ci(rng.normal(0.5, 0.1, 3000), n_boot=500, seed=1)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_paired_test_detects_a_consistent_improvement():
    a = np.linspace(0.2, 0.9, 60) + 0.05
    b = np.linspace(0.2, 0.9, 60)
    result = paired_test(a, b)
    assert result["significant"]
    assert result["p_value"] < 0.05
    assert result["median_diff"] > 0


def test_paired_test_reports_no_difference_for_identical_input():
    a = np.linspace(0.2, 0.9, 40)
    result = paired_test(a, a.copy())
    assert result["p_value"] == 1.0
    assert not result["significant"]
    assert result["n_differing"] == 0


def test_paired_test_is_not_fooled_by_noise():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.1, 100)
    b = rng.normal(0.5, 0.1, 100)
    assert not paired_test(a, b)["significant"]
