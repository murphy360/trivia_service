from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import JobStatus, QuestionType


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=1, description="Prompt/topic to generate questions about")
    category: str = Field(min_length=1)
    difficulty: str = Field(default="medium")
    type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE)
    count: int = Field(default=1, ge=1, le=20)


class QuestionResponse(BaseModel):
    id: int
    type: QuestionType
    category: str
    difficulty: str
    question_text: str
    correct_answer: str
    choices: list[str] | None
    generator_provider: str
    verifier_provider: str
    fact_check_notes: str
    created_at: datetime

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: int
    status: JobStatus
    question_ids: list[int]
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}
