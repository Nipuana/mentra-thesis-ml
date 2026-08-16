"""The split is the foundation every RQ1 number rests on.

If a learner's future leaks into training, every metric in the thesis is
inflated. These tests pin the properties that make the protocol defensible.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.pipelines.evaluate import temporal_split, temporal_three_way


def _history(user: str, courses: list[str], start: str = "2026-01-01") -> pd.DataFrame:
    """One row per course, one day apart, in the given order."""
    ts = pd.date_range(start, periods=len(courses), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "user_id": user,
            "course_id": courses,
            "weight": 1.0,
            "ts": ts,
        }
    )


def test_held_out_items_never_appear_in_training():
    inter = _history("u1", [f"c{i}" for i in range(10)])
    train, test = temporal_split(inter, holdout_frac=0.3)

    train_items = set(train[train["user_id"] == "u1"]["course_id"])
    assert train_items.isdisjoint(test["u1"]), "held-out courses leaked into training"


def test_split_is_temporal_not_random():
    """Everything held out must be strictly newer than everything trained on."""
    inter = _history("u1", [f"c{i}" for i in range(10)])
    train, test = temporal_split(inter, holdout_frac=0.3)

    ts_by_course = dict(zip(inter["course_id"], inter["ts"]))
    newest_train = max(ts_by_course[c] for c in train["course_id"])
    oldest_test = min(ts_by_course[c] for c in test["u1"])
    assert newest_train < oldest_test


def test_holdout_fraction_is_respected():
    inter = _history("u1", [f"c{i}" for i in range(10)])
    _, test = temporal_split(inter, holdout_frac=0.3)
    assert len(test["u1"]) == 3


def test_learner_always_keeps_some_history():
    """A 100% holdout would leave nothing to personalize from."""
    inter = _history("u1", ["a", "b"])
    train, test = temporal_split(inter, holdout_frac=1.0)
    assert len(train) >= 1
    assert len(test["u1"]) >= 1


def test_short_histories_train_but_are_not_scored():
    inter = _history("u1", ["only-one"])
    train, test = temporal_split(inter, holdout_frac=0.3, min_history=2)
    assert "u1" not in test, "a learner with one interaction has nothing to predict"
    assert len(train) == 1, "their row should still inform the models"


def test_unsorted_input_is_ordered_before_splitting():
    """Rows arrive from SQL in arbitrary order; the split must not depend on it."""
    inter = _history("u1", [f"c{i}" for i in range(10)])
    shuffled = inter.iloc[[7, 2, 9, 0, 4, 1, 8, 3, 6, 5]].reset_index(drop=True)

    _, test_ordered = temporal_split(inter, holdout_frac=0.3)
    _, test_shuffled = temporal_split(shuffled, holdout_frac=0.3)
    assert test_ordered["u1"] == test_shuffled["u1"]


def test_three_way_blocks_are_chronological():
    """train < validation < test, so tuning never sees the test future."""
    inter = _history("u1", [f"c{i}" for i in range(20)])
    train, val, test = temporal_three_way(inter, val_frac=0.15, test_frac=0.3)

    ts = dict(zip(inter["course_id"], inter["ts"]))
    newest_train = max(ts[c] for c in train["course_id"])
    val_span = [ts[c] for c in val["u1"]]
    test_span = [ts[c] for c in test["u1"]]

    assert newest_train < min(val_span)
    assert max(val_span) < min(test_span)


def test_three_way_blocks_are_disjoint():
    inter = _history("u1", [f"c{i}" for i in range(20)])
    train, val, test = temporal_three_way(inter)
    train_items = set(train["course_id"])
    assert train_items.isdisjoint(val["u1"])
    assert train_items.isdisjoint(test["u1"])
    assert val["u1"].isdisjoint(test["u1"])


def test_multiple_learners_are_split_independently():
    inter = pd.concat(
        [
            _history("u1", [f"a{i}" for i in range(10)], start="2026-01-01"),
            _history("u2", [f"b{i}" for i in range(6)], start="2026-06-01"),
        ],
        ignore_index=True,
    )
    train, test = temporal_split(inter, holdout_frac=0.5)
    # u2's history is entirely later than u1's, but that must not push u1's
    # interactions into the test block: the cut is per learner.
    assert set(test) == {"u1", "u2"}
    assert len(test["u1"]) == 5
    assert len(test["u2"]) == 3


@pytest.mark.parametrize("n", [3, 5, 8, 13, 21])
def test_three_way_always_leaves_training_data(n: int):
    inter = _history("u1", [f"c{i}" for i in range(n)])
    train, _, test = temporal_three_way(inter)
    assert len(train) >= 1
    assert len(test["u1"]) >= 1
