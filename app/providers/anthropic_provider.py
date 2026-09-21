import json

from anthropic import AsyncAnthropic

from app.core.config import get_settings
from app.models.enums import QuestionType
from app.providers.base import (
    FACT_CHECK_JSON_INSTRUCTIONS,
    Candidate,
    FactCheckResult,
    generation_instructions,
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        settings = get_settings()
        self._model = settings.anthropic_model
        self._client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(
        self,
        topic: str,
        category: str,
        difficulty: str,
        qtype: QuestionType,
        avoid_questions: list[str],
    ) -> Candidate:
        prompt = (
            f"Generate one trivia question.\nTopic: {topic}\nCategory: {category}\n"
            f"Difficulty: {difficulty}\n\n{generation_instructions(qtype, avoid_questions)}"
        )
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        payload = _extract_json(text)

        return Candidate(
            type=qtype,
            category=category,
            difficulty=difficulty,
            topic_prompt=topic,
            question_text=payload["question_text"],
            correct_answer=payload["correct_answer"],
            choices=payload.get("choices"),
            generator_provider=self.name,
        )

    async def fact_check(self, candidate: Candidate) -> FactCheckResult:
        # Server-side tool: Claude runs its own search turns internally within this
        # one API call and returns the final answer, so no manual tool-use loop here.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{FACT_CHECK_JSON_INSTRUCTIONS}\n\nQuestion: {candidate.question_text}\n"
                        f"Claimed correct answer: {candidate.correct_answer}\n"
                        f"Category: {candidate.category}"
                    ),
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        payload = _extract_json(text)
        return FactCheckResult(
            verified=payload["verified"],
            notes=payload["notes"],
            verifier_provider=self.name,
        )
