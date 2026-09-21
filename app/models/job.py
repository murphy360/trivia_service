from datetime import datetime, timezone

from sqlalchemy import Column
from sqlmodel import JSON, Field, SQLModel

from app.models.enums import JobStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GenerationJob(SQLModel, table=True):
    __tablename__ = "generation_jobs"

    id: int | None = Field(default=None, primary_key=True)
    params: dict = Field(sa_column=Column(JSON))
    status: JobStatus = Field(default=JobStatus.PENDING, index=True)
    question_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = Field(default=None)
