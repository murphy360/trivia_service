import random
from typing import Protocol

from pydantic import BaseModel

from app.models.enums import QuestionType


class Candidate(BaseModel):
    """A single generated question, not yet fact-checked or persisted."""

    type: QuestionType
    category: str
    difficulty: str
    topic_prompt: str
    question_text: str
    correct_answer: str
    choices: list[str] | None = None  # multiple_choice only: all options, correct + distractors
    generator_provider: str


class FactCheckResult(BaseModel):
    verified: bool
    notes: str
    verifier_provider: str


# Shared instructions for every provider's generation prompt, so the JSON contract
# (and therefore downstream parsing) doesn't drift between vendors.
GENERATION_JSON_INSTRUCTIONS = """
Respond with ONLY a JSON object (no markdown fences, no commentary) matching this shape:
{{
  "question_text": string,
  "correct_answer": string,
  "choices": array of strings or null
}}

Rules for "{qtype}" questions:
{type_rules}

The question must be non-trivial, unambiguous, and have exactly one defensible correct answer.
""".strip()

_TYPE_RULES = {
    QuestionType.MULTIPLE_CHOICE: (
        '"choices" must contain exactly 4 strings total: the correct answer plus 3 '
        "plausible but definitively incorrect distractors, in any order."
    ),
    QuestionType.TRUE_FALSE: (
        '"question_text" MUST be a single declarative STATEMENT that is either true or '
        'false — never an open question starting with "which", "what", "who", "when", '
        'or "where". For example: "Sushruta, an ancient Indian physician, is credited '
        'with pioneering early techniques of rhinoplasty." (not "Which ancient Indian '
        'physician pioneered rhinoplasty?"). "correct_answer" must be exactly "True" or '
        '"False". "choices" must be null — do not include a list of options.'
    ),
    QuestionType.FILL_IN_BLANK: (
        'The "question_text" must contain a blank shown as "_____". "correct_answer" '
        'is the single word or short phrase that fills it. "choices" must be null — '
        "do not include a list of options."
    ),
    QuestionType.SHORT_ANSWER: (
        '"correct_answer" is a short, canonical answer (a few words at most). '
        '"choices" must be null — do not include a list of options.'
    ),
}


def generation_instructions(qtype: QuestionType, avoid_questions: list[str] | None = None) -> str:
    type_rules = _TYPE_RULES[qtype]
    if qtype == QuestionType.TRUE_FALSE:
        # Decided here, not left to the model: LLMs have a well-documented bias toward
        # writing true statements (it's easier than crafting a plausible false one), so
        # asking for "either True or False" in practice skews heavily toward True. Forcing
        # a specific target per question is what actually guarantees a 50/50 split.
        target = random.choice(["True", "False"])
        type_rules += (
            f'\nFor THIS question specifically, "correct_answer" MUST be "{target}". Write a '
            f"declarative statement whose truth value is {target}"
            + (
                " — an accurate factual claim."
                if target == "True"
                else " — a plausible-sounding claim that is actually incorrect (e.g. swap a "
                "name, date, or number from the true fact)."
            )
        )
    instructions = GENERATION_JSON_INSTRUCTIONS.format(qtype=qtype.value, type_rules=type_rules)
    if avoid_questions:
        bullets = "\n".join(f"- {q}" for q in avoid_questions)
        instructions += (
            "\n\nThe following questions already exist for this category — do not repeat "
            f"or closely paraphrase any of them. Pick a genuinely different fact, angle, "
            f"or sub-topic:\n{bullets}"
        )
    return instructions


FACT_CHECK_JSON_INSTRUCTIONS = """
You are fact-checking a trivia question you did NOT write. Use your web search
capability to independently verify the claim before answering. Respond with ONLY a
JSON object (no markdown fences, no commentary) matching this shape:
{
  "verified": boolean,
  "notes": string
}
"verified" must be false if the question is ambiguous, outdated, opinion-based, or the
stated correct_answer is not factually correct. "notes" should briefly cite what you
found, in one or two sentences.
""".strip()


class Provider(Protocol):
    """Each provider fact-checks using its own native web search grounding (Anthropic's
    web_search server tool, OpenAI's Responses API web_search tool, Gemini's Google
    Search tool) rather than a shared search API — every vendor already ships one, and
    forcing a single external key onto all of them was an unnecessary dependency."""

    name: str

    async def generate(
        self,
        topic: str,
        category: str,
        difficulty: str,
        qtype: QuestionType,
        avoid_questions: list[str],
    ) -> Candidate: ...

    async def fact_check(self, candidate: Candidate) -> FactCheckResult: ...
