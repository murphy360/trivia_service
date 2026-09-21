from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.auth import require_api_key
from app.core.db import get_session
from app.models.enums import QuestionType
from app.repository.questions import list_questions
from app.schemas.generation import QuestionResponse

router = APIRouter(tags=["questions"], dependencies=[Depends(require_api_key)])


@router.get("/questions", response_model=list[QuestionResponse])
async def get_questions(
    category: str | None = None,
    difficulty: str | None = None,
    type: QuestionType | None = None,
    exclude_ids: list[int] = Query(default_factory=list),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[QuestionResponse]:
    questions = await list_questions(
        session,
        category=category,
        difficulty=difficulty,
        type_=type,
        exclude_ids=exclude_ids,
        limit=limit,
    )
    return [QuestionResponse.model_validate(q) for q in questions]
