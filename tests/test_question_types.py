from app.models.enums import QuestionType
from app.pipeline.question_types import validate_candidate
from app.providers.base import Candidate


def _candidate(**overrides) -> Candidate:
    defaults = dict(
        type=QuestionType.MULTIPLE_CHOICE,
        category="geography",
        difficulty="easy",
        topic_prompt="France",
        question_text="What is the capital of France?",
        correct_answer="Paris",
        choices=["Paris", "London", "Berlin", "Madrid"],
        generator_provider="fake",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def test_valid_multiple_choice_has_no_errors():
    assert validate_candidate(_candidate()) == []


def test_multiple_choice_requires_correct_answer_in_choices():
    errors = validate_candidate(_candidate(correct_answer="Rome"))
    assert any("correct_answer must be one of the choices" in e for e in errors)


def test_multiple_choice_requires_exactly_four_choices():
    errors = validate_candidate(_candidate(choices=["Paris", "London"]))
    assert any("exactly 4 choices" in e for e in errors)


def test_valid_true_false_has_no_errors():
    candidate = _candidate(
        type=QuestionType.TRUE_FALSE,
        question_text="Paris is the capital of France.",
        correct_answer="True",
        choices=None,
    )
    assert validate_candidate(candidate) == []


def test_true_false_rejects_non_boolean_answer():
    candidate = _candidate(
        type=QuestionType.TRUE_FALSE,
        question_text="Paris is the capital of France.",
        correct_answer="Yes",
        choices=None,
    )
    errors = validate_candidate(candidate)
    assert any("True" in e for e in errors)


def test_valid_fill_in_blank_has_no_errors():
    candidate = _candidate(
        type=QuestionType.FILL_IN_BLANK,
        question_text="The capital of France is _____.",
        correct_answer="Paris",
        choices=None,
    )
    assert validate_candidate(candidate) == []


def test_fill_in_blank_requires_blank_marker():
    candidate = _candidate(
        type=QuestionType.FILL_IN_BLANK,
        question_text="What is the capital of France?",
        correct_answer="Paris",
        choices=None,
    )
    errors = validate_candidate(candidate)
    assert any("_____" in e for e in errors)


def test_valid_short_answer_has_no_errors():
    candidate = _candidate(
        type=QuestionType.SHORT_ANSWER,
        question_text="What is the capital of France?",
        correct_answer="Paris",
        choices=None,
    )
    assert validate_candidate(candidate) == []
