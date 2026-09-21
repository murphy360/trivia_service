import json

from google import genai
from google.genai import types

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


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: genai.Client | None = None) -> None:
        settings = get_settings()
        self._model = settings.gemini_model
        self._client = client or genai.Client(api_key=settings.gemini_api_key)

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
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        payload = _extract_json(response.text)

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
        # Google Search grounding is a server-side tool: Gemini runs its own search
        # turns internally within this one call, same as the other providers.
        config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=(
                f"{FACT_CHECK_JSON_INSTRUCTIONS}\n\nQuestion: {candidate.question_text}\n"
                f"Claimed correct answer: {candidate.correct_answer}\n"
                f"Category: {candidate.category}"
            ),
            config=config,
        )
        payload = _extract_json(response.text)
        return FactCheckResult(
            verified=payload["verified"],
            notes=payload["notes"],
            verifier_provider=self.name,
        )
