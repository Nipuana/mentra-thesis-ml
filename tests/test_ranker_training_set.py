"""Negative sampling for the ranker's training set.

The sampler oversamples candidates so positives can be filtered out. That
oversampling must stay within the catalogue size — a regression here only shows
up for learners whose history is large relative to the catalogue, which the
seeded dataset never produces but real datasets do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.ranker import FEATURES, build_training_set, course_feature_table


class _StubCF:
    def scores_for_user(self, user_id: str) -> dict[str, float]:
        return {}


def _courses(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "course_id": [f"c{i}" for i in range(n)],
            "enrollment_count": np.arange(n),
            "rating_avg": 4.0,
            "rating_count": 10,
            "duration_minutes": 60,
            "difficulty_level": "beginner",
        }
    )


def _interactions(user: str, item_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"user_id": user, "course_id": item_ids, "weight": 1.0})


def test_builds_positives_and_negatives():
    courses = _courses(50)
    items = courses["course_id"].tolist()
    inter = _interactions("u1", items[:4])

    X, y = build_training_set(inter, _StubCF(), {}, course_feature_table(courses), items)

    assert list(X.columns) == FEATURES
    assert y.sum() == 4, "each interacted course is a positive"
    assert (y == 0).sum() > 0, "negatives are needed to train a classifier"
    assert len(X) == len(y)


def test_history_larger_than_the_oversampling_window():
    """A learner engaged with most of a small catalogue.

    n_neg * 2 + 10 exceeds the catalogue here, so an uncapped sample without
    replacement raises ValueError.
    """
    courses = _courses(30)
    items = courses["course_id"].tolist()
    inter = _interactions("u1", items[:20])

    X, y = build_training_set(inter, _StubCF(), {}, course_feature_table(courses), items)

    assert y.sum() == 20
    assert len(X) == len(y)


def test_negatives_never_include_a_positive():
    courses = _courses(40)
    items = courses["course_id"].tolist()
    positives = items[:6]
    inter = _interactions("u1", positives)

    X, y = build_training_set(inter, _StubCF(), {}, course_feature_table(courses), items)
    # Positives are emitted first, so the rest must be disjoint from them.
    assert y[: len(positives)].all()
    assert not y[len(positives) :].any()


@pytest.mark.parametrize("n_items,n_pos", [(10, 9), (12, 11), (30, 29), (100, 99)])
def test_saturated_catalogues_do_not_raise(n_items: int, n_pos: int):
    """The extreme case: one unseen course left in the whole catalogue."""
    courses = _courses(n_items)
    items = courses["course_id"].tolist()
    inter = _interactions("u1", items[:n_pos])

    X, y = build_training_set(inter, _StubCF(), {}, course_feature_table(courses), items)
    assert len(X) == len(y) >= n_pos


def test_learner_with_every_course_yields_no_negatives():
    courses = _courses(15)
    items = courses["course_id"].tolist()
    inter = _interactions("u1", items)

    X, y = build_training_set(inter, _StubCF(), {}, course_feature_table(courses), items)
    assert y.sum() == len(items)
    assert (y == 0).sum() == 0


def test_empty_interactions_return_empty_frames():
    courses = _courses(10)
    empty = pd.DataFrame({"user_id": [], "course_id": [], "weight": []})

    X, y = build_training_set(empty, _StubCF(), {}, course_feature_table(courses),
                              courses["course_id"].tolist())
    assert len(X) == 0
    assert len(y) == 0
