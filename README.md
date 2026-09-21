# trivia_service

Standalone trivia question generation, fact-checking, and storage service. It takes a
topic/prompt, generates candidate questions across multiple LLM providers in parallel,
independently fact-checks each one with a web-search tool, rejects anything too similar
to a question already in the bank, and persists the survivors to SQLite. It's meant to
be reused by more than one consumer — e.g. a Discord trivia bot and a future "daily
trivia text to friends" service — which just call this service's HTTP API and own their
own gameplay/stats.

This service is built and run entirely through Docker. There is no supported bare-host
Python setup.

## Setup

```bash
cp .env.example .env
# fill in TRIVIA_SERVICE_API_KEY, and at least one of ANTHROPIC_API_KEY /
# OPENAI_API_KEY / GEMINI_API_KEY, plus SEARCH_API_KEY (a Tavily API key) for
# fact-checking.
```

## Run

```bash
docker compose up --build
```

The API is served at `http://localhost:8000`. SQLite data persists in `./data` on the
host via the mounted volume.

## Test

```bash
docker compose run --rm app pytest
```

Tests use fake in-memory providers and a temp SQLite file — no real LLM/search API
calls, no network access required.

## API

All endpoints except `/health` require an `X-API-Key` header matching
`TRIVIA_SERVICE_API_KEY`.

- `POST /generate` — body: `{"topic": str, "category": str, "difficulty": str,
  "type": "multiple_choice" | "true_false" | "fill_in_blank" | "short_answer",
  "count": int}`. Returns `202` with a job. Generation runs as a background task;
  poll `GET /jobs/{id}` for status/result question ids.
- `GET /jobs/{id}` — job status (`pending`/`running`/`completed`/`failed`) and the
  ids of any questions it produced.
- `GET /questions?category=&difficulty=&type=&exclude_ids=&limit=` — fetch stored
  questions, optionally excluding ids a consumer has already served.
- `GET /health` — liveness check, no auth.

## How generation works

For each requested question, a generator provider produces a candidate, then a
*different* provider fact-checks it using a shared web-search tool (so verification
behavior is consistent across vendors). Only candidates that pass fact-checking and
per-type validation are embedded (locally, via sentence-transformers) and compared
against existing questions in the same category; anything too similar
(`NOVELTY_SIMILARITY_THRESHOLD`, default 0.87 cosine similarity) is discarded instead
of stored, so the bank doesn't accumulate near-duplicates.

Providers are enabled purely by which API keys are set in `.env` — Anthropic, OpenAI,
and Gemini adapters ship today; adding another vendor is a new adapter in
`app/providers/` plus one line in `app/providers/registry.py`.

## Adding a question type

Question formats live in `app/pipeline/question_types.py` as validation rules keyed by
`QuestionType`. Adding a new one means adding an enum value in `app/models/enums.py`
and a validation branch there — the pipeline and API don't need to change.
