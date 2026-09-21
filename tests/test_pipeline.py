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
        # Provider pairing is randomized, but a candidate must never verify itself.
        assert {job.attempts[0]["generator_provider"], job.attempts[0]["verifier_provider"]} == {
            "fake-a",
            "fake-b",
        }
        assert job.attempts[0]["generator_provider"] != job.attempts[0]["verifier_provider"]


async def test_all_providers_failing_fact_check_exhausts_retries(monkeypatch):
    # Every provider fails fact-check, so no matter which one is randomly chosen as
    # generator vs. verifier on the retry, the slot exhausts its attempt budget (one
    # try per enabled provider) and gives up rather than looping forever.
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [
            FakeProvider("fake-a", verified=False),
            FakeProvider("fake-b", verified=False),
        ],
    )
    params = GenerateRequest(
        topic="France", category="geography-pipeline-allfail", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=1,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.question_ids == []
        assert len(job.attempts) == 2  # one attempt per enabled provider, then give up
        assert all(a["outcome"] == "fact_check_failed" for a in job.attempts)
        assert all(a["question_id"] is None for a in job.attempts)


async def test_slot_retries_with_a_different_provider_after_fact_check_failure(monkeypatch):
    # Deterministic version of the retry mechanic: force the first pairing to fail
    # fact-check, then confirm the retry actually swaps in a different verifier
    # (rather than discarding after one try) and the second attempt persists.
    generator = FakeProvider("generator")
    bad_verifier = FakeProvider("bad-verifier", verified=False)
    good_verifier = FakeProvider("good-verifier", verified=True)

    calls = iter([(generator, bad_verifier), (generator, good_verifier)])
    monkeypatch.setattr(
        orchestrator, "pick_generator_and_verifier", lambda providers, tried: next(calls)
    )
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [generator, bad_verifier, good_verifier],
    )

    params = GenerateRequest(
        topic="France", category="geography-pipeline-retry", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=1,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)

    async with AsyncSession(engine) as session:
        job = await session.get(GenerationJob, job_id)
        assert job.status == JobStatus.COMPLETED
        assert len(job.question_ids) == 1
        assert len(job.attempts) == 2
        assert job.attempts[0]["outcome"] == "fact_check_failed"
        assert job.attempts[0]["verifier_provider"] == "bad-verifier"
        assert job.attempts[1]["outcome"] == "persisted"
        assert job.attempts[1]["verifier_provider"] == "good-verifier"


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
        # Both fake providers always propose the identical question text, so one slot
        # persists and the other keeps colliding with it on every retry (both enabled
        # providers produce the same text) until it exhausts its attempt budget.
        assert len(job.question_ids) == 1
        outcomes = [a["outcome"] for a in job.attempts]
        assert outcomes.count("persisted") == 1
        assert outcomes.count("duplicate") == 2
        assert len(job.attempts) == 3


async def test_generation_is_steered_away_from_existing_questions(monkeypatch):
    provider_a = FakeProvider("fake-a")
    provider_b = FakeProvider("fake-b")
    monkeypatch.setattr(
        orchestrator, "get_enabled_providers", lambda: [provider_a, provider_b]
    )
    params = GenerateRequest(
        topic="France", category="geography-pipeline-avoid", difficulty="easy",
        type=QuestionType.MULTIPLE_CHOICE, count=1,
    )
    job_id = await _make_job(params)

    await orchestrator.run_generation_job(job_id, params)
    # Whichever provider generated should have been told what's already in the bank
    # for this category (empty, on this first request).
    called = provider_a.avoid_questions_seen + provider_b.avoid_questions_seen
    assert len(called) == 1
    assert called[0] == []

    job_id_2 = await _make_job(params)
    await orchestrator.run_generation_job(job_id_2, params)
    called = provider_a.avoid_questions_seen + provider_b.avoid_questions_seen
    # The second request's generator should now see the first request's persisted
    # question text in its avoid-list.
    assert any("What is the capital of France?" in seen for seen in called if seen)
