from app.pipeline import novelty


def test_embed_to_bytes_round_trip():
    vector = novelty.embed("What is the capital of France?")
    restored = novelty.from_bytes(novelty.to_bytes(vector))
    assert restored.shape == vector.shape
    assert restored.dtype == vector.dtype


def test_is_novel_true_when_no_existing_questions():
    vector = novelty.embed("Anything at all")
    assert novelty.is_novel(vector, [], threshold=0.5) is True


def test_is_novel_true_for_distinct_questions():
    existing = novelty.embed("What is the capital of France?")
    candidate = novelty.embed("Which planet is known as the Red Planet?")
    assert novelty.is_novel(candidate, [existing], threshold=0.9) is True


def test_is_novel_false_for_identical_question():
    existing = novelty.embed("What is the capital of France?")
    candidate = novelty.embed("What is the capital of France?")
    assert novelty.is_novel(candidate, [existing], threshold=0.9) is False
