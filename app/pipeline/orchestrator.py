import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import engine
from app.models.enums import JobStatus
from app.models.job import GenerationJob
from app.pipeline import novelty
from app.pipeline.question_types import validate_candidate
from app.pipeline.types import Attempt
from app.providers.base import Candidate, FactCheckResult, Provider
from app.providers.registry import get_enabled_providers, pick_generator_and_verifier
from app.repository.questions import (
    create_question,
    get_embeddings_for_category,
    get_question_texts_for_category,
)
from app.schemas.generation import GenerateRequest

logger = logging.getLogger(__name__)


async def _generate_and_verify(
    generator: Provider,
    verifier: Provider,
    params: GenerateRequest,
    avoid_questions: list[str],
) -> tuple[Attempt, Candidate | None, FactCheckResult | None]:
    """Runs one candidate through generate -> validate -> fact-check. Always returns
    an Attempt describing what happened, even on failure, so the caller never has to
    guess why a candidate didn't make it — only a successful, verified candidate also
    gets a non-None Candidate/FactCheckResult back, for the novelty check to finish."""

    try:
        candidate = await generator.generate(
            params.topic, params.category, params.difficulty, params.type, avoid_questions
        )
    except Exception as exc:
        logger.exception("Provider %s failed to generate a candidate", generator.name)
        return (
            Attempt(
                generator_provider=generator.name,
                verifier_provider=verifier.name,
                outcome="generation_failed",
                detail=str(exc),
            ),
            None,
            None,
        )

    errors = validate_candidate(candidate)
    if errors:
        logger.info("Discarding malformed candidate from %s: %s", generator.name, errors)
        return (
            Attempt(
                generator_provider=generator.name,
                verifier_provider=verifier.name,
                question_text=candidate.question_text,
                outcome="invalid",
                detail="; ".join(errors),
            ),
            None,
            None,
        )

    try:
        fact_check = await verifier.fact_check(candidate)
    except Exception as exc:
        logger.exception("Provider %s failed to fact-check a candidate", verifier.name)
        return (
            Attempt(
                generator_provider=generator.name,
                verifier_provider=verifier.name,
                question_text=candidate.question_text,
                outcome="fact_check_error",
                detail=str(exc),
            ),
            None,
            None,
        )

    if not fact_check.verified:
        logger.info("Discarding unverified candidate: %s", fact_check.notes)
        return (
            Attempt(
                generator_provider=generator.name,
                verifier_provider=verifier.name,
                question_text=candidate.question_text,
                outcome="fact_check_failed",
                detail=fact_check.notes,
            ),
            None,
            None,
        )

    return (
        Attempt(
            generator_provider=generator.name,
            verifier_provider=verifier.name,
            question_text=candidate.question_text,
            outcome="persisted",  # provisional; the novelty check below may override it
            detail=fact_check.notes,
        ),
        candidate,
        fact_check,
    )


async def _fill_one_slot(
    params: GenerateRequest,
    providers: list[Provider],
    shared_texts: list[str],
    shared_vectors: list[np.ndarray],
    lock: asyncio.Lock,
    settings: Settings,
    session: AsyncSession,
    max_attempts: int,
) -> tuple[list[Attempt], int | None]:
    """One requested question's full lifecycle: try a randomly-chosen provider pair,
    and on ANY failure (generation error, malformed output, failed fact-check, or a
    near-duplicate) retry with a provider that hasn't generated for this slot yet,
    up to once per enabled provider. Every attempt is logged, win or lose."""

    attempts_log: list[Attempt] = []
    tried_generators: set[str] = set()
    # Seeded from what's already stored, then grown with whatever this slot itself
    # tries (rejected or not) so a retry doesn't just re-propose the same question.
    local_avoid = list(shared_texts)

    for _ in range(max_attempts):
        generator, verifier = pick_generator_and_verifier(providers, tried_generators)
        tried_generators.add(generator.name)

        attempt, candidate, fact_check = await _generate_and_verify(
            generator, verifier, params, local_avoid
        )

        if candidate is None:
            attempts_log.append(attempt)
            continue

        local_avoid.append(candidate.question_text)
        vector = novelty.embed(candidate.question_text)

        async with lock:
            similarity = novelty.max_similarity(vector, shared_vectors)
            if similarity >= settings.novelty_similarity_threshold:
                logger.info("Discarding near-duplicate question: %r", candidate.question_text)
                attempt = attempt.model_copy(
                    update={
                        "outcome": "duplicate",
                        "detail": (
                            f"Too similar to an existing question in this category "
                            f"(similarity {similarity:.3f} >= threshold "
                            f"{settings.novelty_similarity_threshold})."
                        ),
                    }
                )
                attempts_log.append(attempt)
                continue

            question = await create_question(
                session, candidate, fact_check, novelty.to_bytes(vector)
            )
            shared_vectors.append(vector)
            shared_texts.append(candidate.question_text)
            attempt = attempt.model_copy(update={"question_id": question.id})
            attempts_log.append(attempt)
            return attempts_log, question.id

    return attempts_log, None


async def run_generation_job(job_id: int, params: GenerateRequest) -> None:
    settings = get_settings()

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        if job is None:
            logger.error("GenerationJob %s not found", job_id)
            return
        job.status = JobStatus.RUNNING
        session.add(job)
        await session.commit()

        try:
            providers = get_enabled_providers()
            # One shot per enabled provider per question slot: a retry always lands
            # on a provider that hasn't already failed this slot, and it's bounded.
            max_attempts = len(providers)

            shared_vectors = await get_embeddings_for_category(session, params.category)
            shared_texts = await get_question_texts_for_category(session, params.category)
            lock = asyncio.Lock()

            results = await asyncio.gather(
                *(
                    _fill_one_slot(
                        params,
                        providers,
                        shared_texts,
                        shared_vectors,
                        lock,
                        settings,
                        session,
                        max_attempts,
                    )
                    for _ in range(params.count)
                )
            )

            attempts: list[dict] = []
            question_ids: list[int] = []
            for slot_attempts, question_id in results:
                attempts.extend(a.model_dump() for a in slot_attempts)
                if question_id is not None:
                    question_ids.append(question_id)

            job.status = JobStatus.COMPLETED
            job.question_ids = question_ids
            job.attempts = attempts
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via job.error
            logger.exception("GenerationJob %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)
            session.add(job)
            await session.commit()
