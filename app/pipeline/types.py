from typing import Literal

from pydantic import BaseModel

AttemptOutcome = Literal[
    "generation_failed",
    "invalid",
    "fact_check_error",
    "fact_check_failed",
    "duplicate",
    "persisted",
]


class Attempt(BaseModel):
    """One candidate's journey through the pipeline, kept for observability —
    which providers were paired, where it stopped, and why, so a caller (or the
    test console) never has to guess why a job produced fewer questions than
    requested."""

    generator_provider: str
    verifier_provider: str
    question_text: str | None = None
    outcome: AttemptOutcome
    detail: str
    question_id: int | None = None
