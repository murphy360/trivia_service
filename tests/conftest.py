import os
import tempfile

# Force (not setdefault): docker-compose loads .env into the container's real
# environment before pytest runs, so these must override it unconditionally —
# otherwise tests silently point at the real API key and the real persistent DB.
os.environ["TRIVIA_SERVICE_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{tempfile.mktemp(suffix='.db')}"

import pytest_asyncio  # noqa: E402

from app.core.db import init_db  # noqa: E402
from app.models.enums import QuestionType  # noqa: E402
from app.providers.base import Candidate, FactCheckResult  # noqa: E402


class FakeProvider:
    """A Provider stand-in with no network calls, for pipeline/API tests."""

    def __init__(
        self,
        name: str,
        question_text: str = "What is the capital of France?",
        correct_answer: str = "Paris",
        verified: bool = True,
    ) -> None:
        self.name = name
        self._question_text = question_text
        self._correct_answer = correct_answer
        self._verified = verified

    async def generate(self, topic, category, difficulty, qtype) -> Candidate:
        choices = None
        if qtype == QuestionType.MULTIPLE_CHOICE:
            choices = [self._correct_answer, "London", "Berlin", "Madrid"]
        return Candidate(
            type=qtype,
            category=category,
            difficulty=difficulty,
            topic_prompt=topic,
            question_text=self._question_text,
            correct_answer=self._correct_answer,
            choices=choices,
            generator_provider=self.name,
        )

    async def fact_check(self, candidate, search_tool) -> FactCheckResult:
        return FactCheckResult(
            verified=self._verified, notes="fake fact-check", verifier_provider=self.name
        )


@pytest_asyncio.fixture(autouse=True)
async def _init_db():
    await init_db()
    yield
