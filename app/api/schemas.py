from pydantic import BaseModel


class RecItem(BaseModel):
    course_id: str
    title: str
    field: str | None = None
    sector: str | None = None
    thumbnail_url: str | None = None
    difficulty: str
    rating_avg: float
    rating_count: int
    enrollment_count: int
    instructor_id: str | None = None
    score: float
    reason: str


class RecResponse(BaseModel):
    items: list[RecItem]
    strategy: str
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    ready: bool
    manifest: dict
