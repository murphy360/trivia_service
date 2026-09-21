from httpx import ASGITransport, AsyncClient

from app.main import app
from app.pipeline import orchestrator
from tests.conftest import FakeProvider

API_KEY = "test-key"


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_generate_requires_api_key():
    async with await _client() as client:
        response = await client.post(
            "/generate",
            json={"topic": "France", "category": "geography-api-auth", "type": "multiple_choice"},
        )
    assert response.status_code == 401


async def test_generate_then_fetch_question(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_enabled_providers",
        lambda: [FakeProvider("fake-a"), FakeProvider("fake-b")],
    )

    async with await _client() as client:
        headers = {"X-API-Key": API_KEY}
        create = await client.post(
            "/generate",
            headers=headers,
            json={
                "topic": "France",
                "category": "geography-api-e2e",
                "type": "multiple_choice",
                "count": 1,
            },
        )
        assert create.status_code == 202
        job_id = create.json()["id"]

        job = await client.get(f"/jobs/{job_id}", headers=headers)
        assert job.status_code == 200
        job_body = job.json()
        assert job_body["status"] == "completed"
        question_ids = job_body["question_ids"]
        assert len(question_ids) == 1
        assert len(job_body["attempts"]) == 1
        assert job_body["attempts"][0]["outcome"] == "persisted"

        questions = await client.get(
            "/questions", headers=headers, params={"category": "geography-api-e2e"}
        )
        assert questions.status_code == 200
        body = questions.json()
        assert len(body) == 1
        assert body[0]["id"] == question_ids[0]
        assert body[0]["correct_answer"] == "Paris"
