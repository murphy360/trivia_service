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
from app.research.search_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, SearchTool


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
        self, topic: str, category: str, difficulty: str, qtype: QuestionType
    ) -> Candidate:
        prompt = (
            f"Generate one trivia question.\nTopic: {topic}\nCategory: {category}\n"
            f"Difficulty: {difficulty}\n\n{generation_instructions(qtype)}"
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

    async def fact_check(self, candidate: Candidate, search_tool: SearchTool) -> FactCheckResult:
        tools = [
            {
                "name": TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "input_schema": TOOL_PARAMETERS,
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
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                payload = _extract_json(text)
                return FactCheckResult(
                    verified=payload["verified"],
                    notes=payload["notes"],
                    verifier_provider=self.name,
                )

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                results = await search_tool.search(block.input["query"])
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps([r.model_dump() for r in results]),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return FactCheckResult(
            verified=False,
            notes="Fact-check did not converge within the tool-use budget.",
            verifier_provider=self.name,
        )
