import json

from openai import AsyncOpenAI

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


class OpenAIProvider:
    name = "openai"

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        settings = get_settings()
        self._model = settings.openai_model
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)

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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _extract_json(response.choices[0].message.content)

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
        # Responses API's hosted web_search tool runs its own search turns server-side
        # within this one call, same as the other providers' native search tools.
        response = await self._client.responses.create(
            model=self._model,
            tools=[{"type": "web_search"}],
            input=(
                f"{FACT_CHECK_JSON_INSTRUCTIONS}\n\nQuestion: {candidate.question_text}\n"
                f"Claimed correct answer: {candidate.correct_answer}\n"
                f"Category: {candidate.category}"
            ),
        )
        payload = _extract_json(response.output_text)
        return FactCheckResult(
            verified=payload["verified"],
            notes=payload["notes"],
            verifier_provider=self.name,
        )
