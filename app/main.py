from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import generate, health, questions
from app.core.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Trivia Service", lifespan=lifespan)

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(questions.router)

# Manual test console for poking the API by hand: http://localhost:8000/ui
# Mounted after the API routers, and under its own prefix, so it can never
# shadow a real endpoint.
app.mount(
    "/ui",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="ui",
)
