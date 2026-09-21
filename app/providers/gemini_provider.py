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
from app.research.search_tool import TOOL_DESCRIPTION, TOOL_NAME, TOOL_PARAMETERS, SearchTool


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
        self, topic: str, category: str, difficulty: str, qtype: QuestionType
    ) -> Candidate:
        prompt = (
            f"Generate one trivia question.\nTopic: {topic}\nCategory: {category}\n"
            f"Difficulty: {difficulty}\n\n{generation_instructions(qtype)}"
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

    async def fact_check(self, candidate: Candidate, search_tool: SearchTool) -> FactCheckResult:
        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=TOOL_NAME,
                    description=TOOL_DESCRIPTION,
                    parameters=TOOL_PARAMETERS,
                )
            ]
        )
        config = types.GenerateContentConfig(tools=[tool])

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            f"{FACT_CHECK_JSON_INSTRUCTIONS}\n\n"
                            f"Question: {candidate.question_text}\n"
                            f"Claimed correct answer: {candidate.correct_answer}\n"
                            f"Category: {candidate.category}"
                        )
                    )
                ],
            )
        ]

        for _ in range(4):  # bounded agentic loop: search, re-search, then must answer
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
            model_content = response.candidates[0].content
            function_calls = [p.function_call for p in model_content.parts if p.function_call]

            if not function_calls:
                payload = _extract_json(response.text)
                return FactCheckResult(
                    verified=payload["verified"],
                    notes=payload["notes"],
                    verifier_provider=self.name,
                )

            contents.append(model_content)
            for call in function_calls:
                results = await search_tool.search(call.args["query"])
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=call.name,
                                response={"results": [r.model_dump() for r in results]},
                            )
                        ],
                    )
                )

        return FactCheckResult(
            verified=False,
            notes="Fact-check did not converge within the tool-use budget.",
            verifier_provider=self.name,
        )
