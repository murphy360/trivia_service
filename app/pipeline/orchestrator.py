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
from app.providers.base import Candidate, FactCheckResult, Provider
from app.providers.registry import get_enabled_providers, pick_generator_verifier_pairs
from app.repository.questions import create_question, get_embeddings_for_category
from app.research.search_tool import SearchTool
from app.schemas.generation import GenerateRequest

logger = logging.getLogger(__name__)


async def _generate_and_verify(
    generator: Provider,
    verifier: Provider,
    params: GenerateRequest,
    search_tool: SearchTool,
) -> tuple[Candidate, FactCheckResult] | None:
    try:
        candidate = await generator.generate(
            params.topic, params.category, params.difficulty, params.type
        )
    except Exception:
        logger.exception("Provider %s failed to generate a candidate", generator.name)
        return None

    errors = validate_candidate(candidate)
    if errors:
        logger.info("Discarding malformed candidate from %s: %s", generator.name, errors)
        return None

    try:
        fact_check = await verifier.fact_check(candidate, search_tool)
    except Exception:
        logger.exception("Provider %s failed to fact-check a candidate", verifier.name)
        return None

    if not fact_check.verified:
        logger.info("Discarding unverified candidate: %s", fact_check.notes)
        return None

    return candidate, fact_check


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
            search_tool = SearchTool()

            outcomes = await asyncio.gather(
                *(
                    _generate_and_verify(generator, verifier, params, search_tool)
                    for generator, verifier in pairs
                )
            )

            existing_vectors = await get_embeddings_for_category(session, params.category)
            question_ids: list[int] = []

            for outcome in outcomes:
                if outcome is None:
                    continue
                candidate, fact_check = outcome

                vector = novelty.embed(candidate.question_text)
                if not novelty.is_novel(
                    vector, existing_vectors, settings.novelty_similarity_threshold
                ):
                    logger.info("Discarding near-duplicate question: %r", candidate.question_text)
                    continue

                question = await create_question(
                    session, candidate, fact_check, novelty.to_bytes(vector)
                )
                existing_vectors.append(vector)
                question_ids.append(question.id)

            job.status = JobStatus.COMPLETED
            job.question_ids = question_ids
        except Exception as exc:  # noqa: BLE001 - surfaced to the client via job.error
            logger.exception("GenerationJob %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)
            session.add(job)
            await session.commit()
