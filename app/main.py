from contextlib import asynccontextmanager

from fastapi import FastAPI

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
