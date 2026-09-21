from app.providers.registry import pick_generator_and_verifier


class _StubProvider:
    def __init__(self, name: str) -> None:
        self.name = name


def test_generator_never_equals_verifier_with_multiple_providers():
    providers = [_StubProvider("a"), _StubProvider("b"), _StubProvider("c")]
    for _ in range(30):
        generator, verifier = pick_generator_and_verifier(providers, set())
        assert generator.name != verifier.name


def test_tried_generators_are_excluded_when_an_alternative_exists():
    providers = [_StubProvider("a"), _StubProvider("b"), _StubProvider("c")]
    generator, verifier = pick_generator_and_verifier(providers, {"a", "b"})
    assert generator.name == "c"
    assert verifier.name != "c"


def test_falls_back_to_full_pool_once_every_provider_has_been_tried():
    providers = [_StubProvider("a"), _StubProvider("b")]
    generator, verifier = pick_generator_and_verifier(providers, {"a", "b"})
    assert generator.name in {"a", "b"}
    assert generator.name != verifier.name


def test_single_provider_must_verify_its_own_work():
    providers = [_StubProvider("solo")]
    generator, verifier = pick_generator_and_verifier(providers, set())
    assert generator.name == verifier.name == "solo"


def test_selection_uses_every_provider_across_many_calls():
    # Sanity check that this is actually randomized, not silently deterministic.
    providers = [_StubProvider("a"), _StubProvider("b"), _StubProvider("c")]
    seen = {pick_generator_and_verifier(providers, set())[0].name for _ in range(50)}
    assert seen == {"a", "b", "c"}
