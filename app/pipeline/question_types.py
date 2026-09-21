from app.models.enums import QuestionType
from app.providers.base import Candidate


def validate_candidate(candidate: Candidate) -> list[str]:
    """Returns a list of validation error strings; empty list means the candidate is
    well-formed for its declared type. This is a cheap, deterministic gate that runs
    *before* the (expensive) novelty check, so malformed output never reaches it."""
    errors: list[str] = []

    if candidate.type == QuestionType.MULTIPLE_CHOICE:
        choices = candidate.choices or []
        if len(choices) != 4:
            errors.append("multiple_choice requires exactly 4 choices")
        if len(set(choices)) != len(choices):
            errors.append("multiple_choice choices must be unique")
        if candidate.correct_answer not in choices:
            errors.append("correct_answer must be one of the choices")

    elif candidate.type == QuestionType.TRUE_FALSE:
        if candidate.choices is not None:
            errors.append("true_false must not have choices")
        if candidate.correct_answer not in ("True", "False"):
            errors.append('true_false correct_answer must be "True" or "False"')

    elif candidate.type == QuestionType.FILL_IN_BLANK:
        if candidate.choices is not None:
            errors.append("fill_in_blank must not have choices")
        if "_____" not in candidate.question_text:
            errors.append("fill_in_blank question_text must contain a _____ blank")
        if not candidate.correct_answer.strip():
            errors.append("fill_in_blank correct_answer must not be empty")

    elif candidate.type == QuestionType.SHORT_ANSWER:
        if candidate.choices is not None:
            errors.append("short_answer must not have choices")
        if not candidate.correct_answer.strip():
            errors.append("short_answer correct_answer must not be empty")

    if not candidate.question_text.strip():
        errors.append("question_text must not be empty")

    return errors
