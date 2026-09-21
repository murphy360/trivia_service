from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.db import engine
from app.models.enums import JobStatus, QuestionType
from app.models.job import GenerationJob
from app.pipeline import orchestrator
from app.schemas.generation import GenerateRequest
from tests.conftest import FakeProvider


async def _make_job(params: GenerateRequest) -> int:
    async with AsyncSession(engine) as session:
        job = GenerationJob(params=params.model_dump(mode="json"))
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job.id


async def test_verified_candidate_is_persisted(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [FakeProvider("fake-a"), FakeProvider("fake-b")],
    )
    params = GenerateRequest(
        topic="France", category="geography-pipeline-verified", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=1,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        assert job.status == JobStatus.COMPLETED
        assert len(job.question_ids) == 1
        assert len(job.attempts) == 1
        assert job.attempts[0]["outcome"] == "persisted"
        assert job.attempts[0]["question_id"] == job.question_ids[0]
        assert job.attempts[0]["generator_provider"] == "fake-a"
        assert job.attempts[0]["verifier_provider"] == "fake-b"


async def test_unverified_candidate_is_discarded(monkeypatch):
    # count=1 pairs providers[0] as generator and providers[1] as verifier, so the
    # *second* provider's `verified` flag is what determines fact-check outcome.
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [FakeProvider("fake-a"), FakeProvider("fake-b", verified=False)],
    )
    params = GenerateRequest(
        topic="France", category="geography-pipeline-unverified", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=1,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.question_ids == []
        assert len(job.attempts) == 1
        assert job.attempts[0]["outcome"] == "fact_check_failed"
        assert job.attempts[0]["detail"] == "fake fact-check"
        assert job.attempts[0]["question_id"] is None


async def test_near_duplicate_second_candidate_is_rejected(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [FakeProvider("fake-a"), FakeProvider("fake-b")],
    )
    params = GenerateRequest(
        topic="France", category="geography-pipeline-dedup", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=2,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        assert job.status == JobStatus.COMPLETED
        # Both candidates ask the identical question, so only one should survive novelty check.
        assert len(job.question_ids) == 1
        assert len(job.attempts) == 2
        outcomes = sorted(a["outcome"] for a in job.attempts)
        assert outcomes == ["duplicate", "persisted"]
