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
from app.research.search_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, SearchTool


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
        self, topic: str, category: str, difficulty: str, qtype: QuestionType
    ) -> Candidate:
        prompt = (
            f"Generate one trivia question.\nTopic: {topic}\nCategory: {category}\n"
            f"Difficulty: {difficulty}\n\n{generation_instructions(qtype)}"
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

    async def fact_check(self, candidate: Candidate, search_tool: SearchTool) -> FactCheckResult:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_PARAMETERS,
                },
            }
        ]
        messages = [
            {
                "role": "user",
                "content": (
                    f"{FACT_CHECK_JSON_INSTRUCTIONS}\n\nQuestion: {candidate.question_text}\n"
                    f"Claimed correct answer: {candidate.correct_answer}\n"
                    f"Category: {candidate.category}"
                ),
            }
        ]

        for _ in range(4):  # bounded agentic loop: search, re-search, then must answer
            response = await self._client.chat.completions.create(
                model=self._model,
                tools=tools,
                messages=messages,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                payload = _extract_json(message.content)
                return FactCheckResult(
                    verified=payload["verified"],
                    notes=payload["notes"],
                    verifier_provider=self.name,
                )

            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                args = json.loads(call.function.arguments)
                results = await search_tool.search(args["query"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps([r.model_dump() for r in results]),
                    }
                )

        return FactCheckResult(
            verified=False,
            notes="Fact-check did not converge within the tool-use budget.",
            verifier_provider=self.name,
        )
