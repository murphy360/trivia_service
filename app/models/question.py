from datetime import datetime, timezone

from sqlalchemy import Column, LargeBinary
from sqlmodel import JSON, Field, SQLModel

from app.models.enums import QuestionType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Question(SQLModel, table=True):
    __tablename__ = "questions"

    id: int | None = Field(default=None, primary_key=True)
    type: QuestionType = Field(index=True)
    category: str = Field(index=True)
    difficulty: str = Field(index=True)
    topic_prompt: str

    question_text: str
    correct_answer: str
    # multiple_choice: list of distractor strings; other types: null
    choices: list[str] | None = Field(default=None, sa_column=Column(JSON))

    # serialized float32 embedding vector, used for the novelty/dedup check
    embedding: bytes = Field(sa_column=Column(LargeBinary))

    generator_provider: str
    verifier_provider: str
    fact_check_notes: str

    created_at: datetime = Field(default_factory=_utcnow)
