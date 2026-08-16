"""Online recommender: loads trained artifacts and serves recommendations.

Flow:  candidate generation (CF + content-profile + popularity)
   ->  scoring (LightGBM ranker if present, else weighted ensemble blend)
   ->  business rules (exclude already-engaged, published-only)
   ->  top-K with a human-readable reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models import ranker as ranker_mod
from app.registry import store


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _join_fields(fs: list[str], quote: bool = True) -> str:
    """Human-friendly join: ['A'] -> A; ['A','B'] -> A and B; longer -> Oxford list."""
    q = [f"'{f}'" if quote else f for f in fs]
    if len(q) == 1:
        return q[0]
    if len(q) == 2:
        return f"{q[0]} and {q[1]}"
    return ", ".join(q[:-1]) + f", and {q[-1]}"


class Recommender:
    def __init__(self) -> None:
        self.content = store.load("content")
        self.cf = store.load("cf")
        self.courses: pd.DataFrame = store.load("courses")
        self.user_profiles: dict[str, dict[str, float]] = store.load("user_profiles")
        # The ranker artifact is always loaded when present so it can be
        # inspected and evaluated; `USE_RANKER` decides whether it scores live
        # traffic (see the note on that setting for the evidence).
        self.ranker_artifact = store.load("ranker") if store.exists("ranker") else None
        self.ranker = self.ranker_artifact if settings.USE_RANKER else None
        self.manifest = store.read_manifest()

        self.course_feat = ranker_mod.course_feature_table(self.courses)
        self.meta = {row["course_id"]: row for row in self.courses.to_dict("records")}

        pop = self.courses.sort_values(
            ["enrollment_count", "rating_avg", "rating_count"], ascending=False
        )
        self.pop_ranked: list[str] = pop["course_id"].tolist()
        self.pop_scores: dict[str, float] = {
            r["course_id"]: float(r["enrollment_count"]) for _, r in self.courses.iterrows()
        }

    # ---------- public API ----------
    def recommend(self, user_id: str, k: int = settings.DEFAULT_K) -> list[dict]:
        profile = self.user_profiles.get(user_id, {})
        seen = set(profile.keys())
        cold = not self.cf.knows_user(user_id) and not profile
        if cold:
            return self._decorate(self.pop_ranked[:k], reason="Popular right now")

        # --- candidate generation ---
        cf_recs = self.cf.recommend(user_id, settings.CAND_PER_SOURCE, exclude=seen)
        cf_ids = [c for c, _ in cf_recs]
        content_scores = self.content.profile_scores(profile) if profile else {}
        content_ids = [c for c, _ in sorted(content_scores.items(), key=lambda x: -x[1]) if c not in seen][
            : settings.CAND_PER_SOURCE
        ]
        pop_ids = [c for c in self.pop_ranked if c not in seen][: settings.CAND_PER_SOURCE]
        cand = list(dict.fromkeys(cf_ids + content_ids + pop_ids))  # union, order-preserving
        cand = [c for c in cand if c in self.meta and c not in seen]
        if not cand:
            return self._decorate(pop_ids[:k], reason="Popular right now")

        cf_scores_full = self.cf.scores_for_user(user_id)

        # --- scoring ---
        if self.ranker is not None:
            X = ranker_mod.feature_frame(cand, cf_scores_full, content_scores, self.course_feat)
            probs = self.ranker.score(X)
            score = dict(zip(cand, probs))
        else:
            cf_n = _minmax({c: cf_scores_full.get(c, 0.0) for c in cand})
            ct_n = _minmax({c: content_scores.get(c, 0.0) for c in cand})
            pop_n = _minmax({c: self.pop_scores.get(c, 0.0) for c in cand})
            score = {
                c: settings.BLEND_CF * cf_n[c]
                + settings.BLEND_CONTENT * ct_n.get(c, 0.0)
                + settings.BLEND_POPULARITY * pop_n[c]
                for c in cand
            }

        ranked = sorted(cand, key=lambda c: -score[c])[:k]

        cf_set, content_set = set(cf_ids[:20]), set(content_ids[:20])
        results = []
        for c in ranked:
            if c in content_set:
                reason = f"Because you're into {self.meta[c]['sector_name']}"
            elif c in cf_set:
                reason = "Learners like you also took this"
            else:
                reason = "Popular pick for you"
            results.append(self._one(c, float(score[c]), reason))
        return results

    def similar(self, course_id: str, k: int = settings.DEFAULT_K) -> list[dict]:
        pairs = self.content.similar(course_id, k)
        return [self._one(cid, s, "Similar content") for cid, s in pairs if cid in self.meta]

    def search(self, query: str, k: int = 20) -> list[dict]:
        scores = self.content.query_scores(query)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        out = [self._one(c, s, "Search match") for c, s in ranked if s > 0.02 and c in self.meta]
        return out[:k]

    def trending(self, k: int = settings.DEFAULT_K) -> list[dict]:
        return self._decorate(self.pop_ranked[:k], reason="Trending now")

    def sectors(self) -> list[str]:
        vals = {m.get("sector_name") for m in self.meta.values() if m.get("sector_name")}
        return sorted(vals)

    def simulate_new_student(
        self, fields: list[str], k: int = settings.DEFAULT_K, seed_rng: int | None = None
    ) -> dict | None:
        """Cold-start what-if: given ONLY one or more declared interests in
        specialized `fields` (e.g. ["React", "Ethical Hacking"]), what would the
        model recommend to a brand-new student (no history)?

        Each call samples a *different* plausible new-student persona: for every
        declared interest we draw a small, popularity-weighted random sample of
        that field's courses, so all the interests are represented in the taste
        profile. The content model then generalizes to similar courses, blended
        with a popularity prior. This mirrors onboarding, where a new learner
        picks a few interests and immediately gets a personalized feed. Pass
        `seed_rng` to reproduce a persona.
        """
        by_field: dict[str, list[str]] = {}
        for c, m in self.meta.items():
            fn = m.get("field_name") or ""
            if fn:
                by_field.setdefault(fn, []).append(c)
        # de-dupe while keeping order, and drop unknown fields
        valid_fields = [f for f in dict.fromkeys(fields) if by_field.get(f)]
        if not valid_fields:
            return None
        field_set = set(valid_fields)

        rng = np.random.default_rng(seed_rng)

        # --- Step 1/2: seed the persona with courses from EACH declared interest
        # so the taste profile reflects all of them (popularity-weighted draw).
        per_field = 2 if len(valid_fields) <= 2 else 1
        seed: list[str] = []
        for f in valid_fields:
            pool = sorted(by_field[f], key=lambda c: -self.pop_scores.get(c, 0.0))[:12]
            w = np.array([self.pop_scores.get(c, 0.0) for c in pool], dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(pool))
            w = w / w.sum()
            take = int(min(len(pool), per_field))
            seed.extend(str(c) for c in rng.choice(pool, size=take, replace=False, p=w))
        seed = list(dict.fromkeys(seed))  # de-dupe, keep order
        n_seed = len(seed)
        profile = {c: 1.0 for c in seed}

        # Persona labels for display (modal difficulty + parent sector of the seeds).
        def _modal(attr: str) -> str | None:
            vals = [self.meta[c].get(attr) for c in seed if self.meta[c].get(attr)]
            return max(set(vals), key=vals.count) if vals else None

        difficulty_lean = _modal("difficulty_level")
        parent_sector = _modal("sector_name")

        # --- Step 3: content model scores every other course against the seed profile.
        content_scores = self.content.profile_scores(profile) if profile else {}
        cand = [c for c in self.meta if c not in seed]
        ct_n = _minmax({c: content_scores.get(c, 0.0) for c in cand})
        pop_n = _minmax({c: self.pop_scores.get(c, 0.0) for c in cand})

        content_top = sorted(cand, key=lambda c: -ct_n.get(c, 0.0))[:4]
        content_matches = [
            {
                "title": self.meta[c]["title"],
                "sector": self.meta[c].get("field_name"),
                "score": round(ct_n.get(c, 0.0), 3),
            }
            for c in content_top
        ]

        # --- Step 4: blend content (declared interest) with a popularity prior.
        w_content, w_pop = 0.75, 0.25
        score = {c: w_content * ct_n.get(c, 0.0) + w_pop * pop_n.get(c, 0.0) for c in cand}
        # --- Step 5: rank and take top-K.
        ranked = sorted(cand, key=lambda c: -score[c])[:k]

        label = _join_fields(valid_fields)
        items = []
        for c in ranked:
            mfield = self.meta[c].get("field_name")
            reason = (
                f"Matches interest in {mfield}"
                if mfield in field_set
                else f"Cross-over from {mfield}"
            )
            items.append(self._one(c, float(score[c]), reason))

        counts: dict[str, int] = {}
        for c in ranked:
            f = self.meta[c].get("field_name") or "Other"
            counts[f] = counts.get(f, 0) + 1
        by_field_counts = sorted(
            ({"field": f, "count": n} for f, n in counts.items()),
            key=lambda x: -x["count"],
        )
        in_field = sum(1 for c in ranked if self.meta[c].get("field_name") in field_set)
        seed_titles = [self.meta[c]["title"] for c in seed]
        breakdown = {
            "in_field": in_field,
            "total": len(ranked),
            "by_field": by_field_counts,
            "seed_titles": seed_titles,
        }
        persona = {
            "parent_sector": parent_sector,
            "difficulty_lean": difficulty_lean,
            "n_seed": n_seed,
            "seed_titles": seed_titles,
        }

        interests_word = "interest" if len(valid_fields) == 1 else "interests"
        trace = [
            {
                "key": "declare",
                "title": f"New student declares {interests_word}",
                "how": f"A brand-new learner (no history) picks {label}.",
                "stat": (
                    (f"{difficulty_lean} level" if difficulty_lean else "cold start")
                    + (f" · {parent_sector}" if parent_sector else "")
                ),
                "chips": valid_fields,
            },
            {
                "key": "seed",
                "title": "Seed a synthetic taste profile",
                "how": (
                    f"Sample {n_seed} popular courses across {len(valid_fields)} {interests_word} "
                    "(popularity-weighted) as the learner's implicit starting point, a different draw each run."
                ),
                "stat": f"{n_seed} seed courses",
                "chips": seed_titles,
            },
            {
                "key": "content",
                "title": "Content model scores the catalog",
                "how": (
                    "Average the seed courses' TF-IDF vectors into one taste vector, then "
                    f"cosine-compare it against all {len(cand)} other courses."
                ),
                "stat": f"{len(cand)} courses scored",
                "matches": content_matches,
            },
            {
                "key": "blend",
                "title": "Blend with a popularity prior",
                "how": "score = 0.75 · content-match + 0.25 · popularity.",
                "stat": f"{int(w_content * 100)}% interest / {int(w_pop * 100)}% popularity",
            },
            {
                "key": "rank",
                "title": "Rank and return the top-K",
                "how": f"Sort by blended score and keep the best {len(ranked)}.",
                "stat": f"{in_field}/{len(ranked)} in-field · {len(ranked) - in_field} cross-over",
            },
        ]

        return {
            "items": items,
            "breakdown": breakdown,
            "persona": persona,
            "trace": trace,
            "fields": valid_fields,
            "field": _join_fields(valid_fields, quote=False),
        }

    # ---------- helpers ----------
    def _one(self, course_id: str, score: float, reason: str) -> dict:
        m = self.meta[course_id]
        return {
            "course_id": course_id,
            "title": m["title"],
            "field": m["field_name"],
            "sector": m["sector_name"],
            "thumbnail_url": m["thumbnail_url"],
            "difficulty": m["difficulty_level"],
            "rating_avg": float(m["rating_avg"]),
            "rating_count": int(m["rating_count"]),
            "enrollment_count": int(m["enrollment_count"]),
            "instructor_id": m["instructor_id"],
            "score": round(float(score), 4),
            "reason": reason,
        }

    def _decorate(self, ids: list[str], reason: str) -> list[dict]:
        return [self._one(c, self.pop_scores.get(c, 0.0), reason) for c in ids if c in self.meta]
