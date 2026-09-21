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
# OPENAI_API_KEY / GEMINI_API_KEY. Fact-checking uses each provider's own native
# web search, so no separate search API key is needed.
```

## Run

```bash
docker compose up --build
```

The API is served at `http://localhost:8000`. SQLite data persists in `./data` on the
host via the mounted volume.

A manual test console is served at `http://localhost:8000/ui` — paste in your
`TRIVIA_SERVICE_API_KEY`, submit a generation request, and browse stored questions,
without needing curl/Postman. It's a debugging tool, not a product surface.

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
- `GET /jobs/{id}` — job status (`pending`/`running`/`completed`/`failed`), the ids
  of any questions it produced, and an `attempts` list covering every candidate the
  pipeline tried — which providers were paired, where each one stopped
  (`generation_failed` / `invalid` / `fact_check_error` / `fact_check_failed` /
  `duplicate` / `persisted`), and why.
- `GET /questions?category=&difficulty=&type=&exclude_ids=&limit=` — fetch stored
  questions, optionally excluding ids a consumer has already served.
- `GET /health` — liveness check, no auth.

## How generation works

For each requested question, a generator provider is chosen at random and produces a
candidate; a *different* randomly-chosen provider fact-checks it using its own native
web search grounding (Anthropic's `web_search` server tool, OpenAI's Responses API
`web_search` tool, Gemini's Google Search tool) — a provider never verifies its own
work, and which two providers get paired varies request to request.

If a candidate fails for any reason — generation error, malformed output, failed
fact-check, or too similar to an existing question — that question is retried with a
provider that hasn't already generated for it, up to once per enabled provider, before
the pipeline gives up on it. Every attempt, successful or not, is recorded in the
job's `attempts` list (see the `/jobs/{id}` entry above) and rendered in the `/ui` test
console, so a failed slot shows exactly which providers were tried and why each one
was rejected.

Only candidates that pass fact-checking and per-type validation are embedded (locally,
via sentence-transformers) and compared against existing questions in the same
category; anything too similar (`NOVELTY_SIMILARITY_THRESHOLD`, default 0.87 cosine
similarity) is discarded instead of stored. To cut down on hitting that rejection in
the first place, every generation prompt is also fed the text of recent questions
already stored in that category and told to pick a genuinely different fact or angle —
steering toward novelty up front rather than relying only on the after-the-fact
similarity check.

Providers are enabled purely by which API keys are set in `.env` — Anthropic, OpenAI,
and Gemini adapters ship today; adding another vendor is a new adapter in
`app/providers/` plus one line in `app/providers/registry.py`.

## Adding a question type

Question formats live in `app/pipeline/question_types.py` as validation rules keyed by
`QuestionType`. Adding a new one means adding an enum value in `app/models/enums.py`
and a validation branch there — the pipeline and API don't need to change.
