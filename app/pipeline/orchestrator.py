import asyncio
import logging
from datetime import datetime, timezone

from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import get_settings
from app.core.db import engine
from app.models.enums import JobStatus
from app.models.job import GenerationJob
from app.pipeline import novelty
from app.pipeline.question_types import validate_candidate
from app.pipeline.types import Attempt
from app.providers.base import Candidate, FactCheckResult, Provider
from app.providers.registry import get_enabled_providers, pick_generator_verifier_pairs
from app.repository.questions import create_question, get_embeddings_for_category
from app.schemas.generation import GenerateRequest

logger = logging.getLogger(__name__)


async def _generate_and_verify(
    generator: Provider,
    verifier: Provider,
    params: GenerateRequest,
) -> tuple[Attempt, Candidate | None, FactCheckResult | None]:
    """Runs one candidate through generate -> validate -> fact-check. Always returns
    an Attempt describing what happened, even on failure, so the caller never has to
    guess why a candidate didn't make it — only a successful, verified candidate also
    gets a non-None Candidate/FactCheckResult back, for the novelty check to finish."""

    try:
        candidate = await generator.generate(
            params.topic, params.category, params.difficulty, params.type
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
            pairs = pick_generator_verifier_pairs(providers, params.count)

            outcomes = await asyncio.gather(
                *(
                    _generate_and_verify(generator, verifier, params)
                    for generator, verifier in pairs
                )
            )

            existing_vectors = await get_embeddings_for_category(session, params.category)
            question_ids: list[int] = []
            attempts: list[dict] = []

            for attempt, candidate, fact_check in outcomes:
                if candidate is None:
                    attempts.append(attempt.model_dump())
                    continue

                vector = novelty.embed(candidate.question_text)
                similarity = novelty.max_similarity(vector, existing_vectors)
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
                    attempts.append(attempt.model_dump())
                    continue

                question = await create_question(
                    session, candidate, fact_check, novelty.to_bytes(vector)
                )
                existing_vectors.append(vector)
                question_ids.append(question.id)
                attempt = attempt.model_copy(update={"question_id": question.id})
                attempts.append(attempt.model_dump())

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
