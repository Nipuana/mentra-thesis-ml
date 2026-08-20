"""The recommendation reason must follow the ranking, not the candidate list.

These tests exist because the explanation shown to a learner is an ethical
commitment (Chapter 13), not a cosmetic label. An explanation that names a
signal which did not drive the ranking is a rationalisation, and is arguably
worse than showing nothing, because it invites trust the mechanism has not
earned. The properties below are what make the commitment checkable.
"""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.serving.recommender import Recommender


@pytest.fixture
def rec() -> Recommender:
    """A Recommender with no trained artifacts.

    `_attribute` and `_reason_text` depend only on the blend weights and the
    course metadata, so bypassing __init__ avoids needing a fitted model on disk.
    """
    r = Recommender.__new__(Recommender)
    r.meta = {
        "c1": {"sector_name": "Technology"},
        "c2": {"sector_name": "Design"},
    }
    return r


def test_attributes_to_the_largest_weighted_contribution(rec: Recommender) -> None:
    """Content wins only when its weighted term actually exceeds the others."""
    # Content is near its maximum, collaborative near its minimum.
    source, share = rec._attribute("c1", cf_n={"c1": 0.05}, ct_n={"c1": 1.0}, pop_n={"c1": 0.0})
    assert source == "content"
    assert share > 0.5


def test_high_raw_content_score_does_not_beat_the_cf_weight(rec: Recommender) -> None:
    """The regression this suite exists for.

    Collaborative carries 0.65 of the blend and content 0.25, so a course that
    scores moderately on both is driven by collaborative filtering. The previous
    implementation labelled such a course "Because you're into X" purely because
    it appeared in the content candidate list, crediting the smaller term.
    """
    assert settings.BLEND_CF > settings.BLEND_CONTENT  # premise of the test
    source, _ = rec._attribute("c1", cf_n={"c1": 0.9}, ct_n={"c1": 0.9}, pop_n={"c1": 0.9})
    assert source == "collaborative"


def test_popularity_is_named_when_nothing_personal_drove_the_rank(rec: Recommender) -> None:
    source, share = rec._attribute("c1", cf_n={"c1": 0.0}, ct_n={"c1": 0.0}, pop_n={"c1": 1.0})
    assert source == "popularity"
    assert share == pytest.approx(1.0)


def test_share_is_a_real_proportion_of_the_blended_score(rec: Recommender) -> None:
    """The share is what makes the attribution auditable, so it must be honest."""
    cf, ct, pop = 1.0, 1.0, 1.0
    source, share = rec._attribute("c1", cf_n={"c1": cf}, ct_n={"c1": ct}, pop_n={"c1": pop})
    expected = (settings.BLEND_CF * cf) / (
        settings.BLEND_CF * cf + settings.BLEND_CONTENT * ct + settings.BLEND_POPULARITY * pop
    )
    assert source == "collaborative"
    assert share == pytest.approx(expected, abs=1e-3)


def test_missing_scores_do_not_raise(rec: Recommender) -> None:
    """A course present in one candidate list but absent from another map."""
    source, share = rec._attribute("c1", cf_n={}, ct_n={}, pop_n={})
    assert source in {"collaborative", "content", "popularity"}
    assert share == 0.0


def test_reason_text_names_the_sector_only_for_content(rec: Recommender) -> None:
    assert "Technology" in rec._reason_text("c1", "content")
    # Naming a sector on a collaborative recommendation would imply the topic
    # drove the rank when the learner's neighbours did.
    assert "Technology" not in rec._reason_text("c1", "collaborative")
    assert "Technology" not in rec._reason_text("c1", "popularity")


@pytest.mark.parametrize("source", ["collaborative", "content", "popularity"])
def test_every_source_produces_a_non_empty_reason(rec: Recommender, source: str) -> None:
    assert rec._reason_text("c1", source).strip()
