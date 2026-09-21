import numpy as np
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.enums import QuestionType
from app.models.question import Question
from app.pipeline.novelty import from_bytes
from app.providers.base import Candidate, FactCheckResult


async def get_embeddings_for_category(session: AsyncSession, category: str) -> list[np.ndarray]:
    result = await session.exec(select(Question.embedding).where(Question.category == category))
    return [from_bytes(blob) for blob in result.all()]


async def get_question_texts_for_category(
    session: AsyncSession, category: str, limit: int = 40
) -> list[str]:
    """Recent question text in this category, fed back into generation prompts so a
    provider can steer away from what's already been asked instead of relying only on
    the post-hoc embedding similarity check to catch duplicates."""
    result = await session.exec(
        select(Question.question_text)
        .where(Question.category == category)
        .order_by(Question.created_at.desc())
        .limit(limit)
    )
    return list(result.all())


async def create_question(
    session: AsyncSession,
    candidate: Candidate,
    fact_check: FactCheckResult,
    embedding_bytes: bytes,
) -> Question:
    question = Question(
        type=candidate.type,
        category=candidate.category,
        difficulty=candidate.difficulty,
        topic_prompt=candidate.topic_prompt,
        question_text=candidate.question_text,
        correct_answer=candidate.correct_answer,
        choices=candidate.choices,
        embedding=embedding_bytes,
        generator_provider=candidate.generator_provider,
        verifier_provider=fact_check.verifier_provider,
        fact_check_notes=fact_check.notes,
    )
    session.add(question)
    await session.commit()
    await session.refresh(question)
    return question


async def list_questions(
    session: AsyncSession,
    category: str | None = None,
    difficulty: str | None = None,
    type_: QuestionType | None = None,
    exclude_ids: list[int] | None = None,
    limit: int = 50,
) -> list[Question]:
    query = select(Question)
    if category is not None:
        query = query.where(Question.category == category)
    if difficulty is not None:
        query = query.where(Question.difficulty == difficulty)
    if type_ is not None:
        query = query.where(Question.type == type_)
    if exclude_ids:
        query = query.where(Question.id.not_in(exclude_ids))
    query = query.order_by(Question.created_at.desc()).limit(limit)

    result = await session.exec(query)
    return list(result.all())
