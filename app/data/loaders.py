"""Read-only loaders that pull the training-ready views out of Postgres.

The ML service only READS these tables; it never writes to business tables.
"""

import pandas as pd

from app.core.config import settings
from app.core.db import read_sql

# --- Course catalogue + popularity stats (published only) ---
_COURSES_SQL = """
SELECT c.id::text                      AS course_id,
       c.title,
       c.headline,
       c.description,
       c.category_id::text             AS category_id,
       c.instructor_id::text           AS instructor_id,
       c.difficulty_level::text        AS difficulty_level,
       c.thumbnail_url,
       c.duration_minutes,
       c.created_at,
       f.name                          AS field_name,
       s.name                          AS sector_name,
       COALESCE(en.cnt, 0)             AS enrollment_count,
       COALESCE(rv.avg_rating, 0)::float AS rating_avg,
       COALESCE(rv.cnt, 0)             AS rating_count
FROM courses c
JOIN categories f ON f.id = c.category_id
LEFT JOIN categories s ON s.id = f.parent_category_id
LEFT JOIN (SELECT course_id, count(*) cnt FROM enrollments GROUP BY course_id) en
       ON en.course_id = c.id
LEFT JOIN (SELECT course_id, avg(rating) avg_rating, count(*) cnt FROM reviews GROUP BY course_id) rv
       ON rv.course_id = c.id
WHERE c.status = 'published'
"""

# --- Per-course text bag (title + headline + description + taxonomy + transcripts) ---
_TEXT_SQL = """
SELECT c.id::text AS course_id,
       c.title || ' ' || COALESCE(c.headline, '') || ' ' || COALESCE(c.description, '')
         || ' ' || COALESCE(f.name, '') || ' ' || COALESCE(s.name, '')
         || ' ' || COALESCE(lt.txt, '') AS text
FROM courses c
JOIN categories f ON f.id = c.category_id
LEFT JOIN categories s ON s.id = f.parent_category_id
LEFT JOIN (
    SELECT course_id, string_agg(title || ' ' || COALESCE(transcript, ''), ' ') AS txt
    FROM lessons GROUP BY course_id
) lt ON lt.course_id = c.id
WHERE c.status = 'published'
"""

# --- Weighted implicit-feedback interactions (has user_id — the user's own data) ---
# `ts` is the most recent moment the user acted on the course. Evaluation splits
# on it so the held-out set is strictly in the future relative to training,
# rather than a random sample of each user's history.
_INTERACTIONS_SQL = """
WITH signals AS (
    SELECT user_id::text AS uid, course_id::text AS cid,
           (:w_enroll + :w_completion * (completion_percent / 100.0)) AS w,
           enrolled_at AS ts
    FROM enrollments
    UNION ALL
    SELECT user_id::text, course_id::text, :w_review * (rating / 5.0), created_at FROM reviews
    UNION ALL
    SELECT user_id::text, course_id::text, :w_wishlist, created_at FROM wishlist_items
)
SELECT uid AS user_id, cid AS course_id, SUM(w)::float AS weight, MAX(ts) AS ts
FROM signals
GROUP BY uid, cid
"""


def load_courses() -> pd.DataFrame:
    return read_sql(_COURSES_SQL)


def load_course_text() -> pd.DataFrame:
    return read_sql(_TEXT_SQL)


def load_interactions() -> pd.DataFrame:
    return read_sql(
        _INTERACTIONS_SQL,
        {
            "w_enroll": settings.W_ENROLL,
            "w_completion": settings.W_COMPLETION,
            "w_review": settings.W_REVIEW,
            "w_wishlist": settings.W_WISHLIST,
        },
    )


# --- Declared/evolved interests expanded to weak (user, course) signals ---
_INTEREST_SIGNAL_SQL = """
WITH ranked AS (
    SELECT ui.user_id::text AS uid, c.id::text AS cid, ui.weight AS w,
           row_number() OVER (
               PARTITION BY ui.user_id, ui.category_id
               ORDER BY COALESCE(en.n, 0) DESC
           ) AS rn
    FROM user_interests ui
    JOIN categories f ON (f.id = ui.category_id OR f.parent_category_id = ui.category_id)
    JOIN courses c ON c.category_id = f.id AND c.status = 'published'
    LEFT JOIN (SELECT course_id, count(*) n FROM enrollments GROUP BY course_id) en
           ON en.course_id = c.id
)
SELECT uid AS user_id, cid AS course_id, (w * :factor)::float AS weight
FROM ranked
WHERE rn <= :top_n
"""


def load_interest_signals(top_n: int = 5, factor: float = 0.6) -> pd.DataFrame:
    """Turn each user's interest categories into weak positive signals toward the
    top courses in those categories, so the trained model reflects declared
    interests (and their drift) — not just clicks."""
    return read_sql(_INTEREST_SIGNAL_SQL, {"top_n": top_n, "factor": factor})
