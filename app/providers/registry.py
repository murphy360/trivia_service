from app.core.config import get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import Provider
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider


def get_enabled_providers() -> list[Provider]:
    """Providers are enabled purely by whether their API key is configured, so adding
    a new vendor later is just: write an adapter, add it here, set its env var."""
    settings = get_settings()
    providers: list[Provider] = []
    if settings.anthropic_api_key:
        providers.append(AnthropicProvider())
    if settings.openai_api_key:
        providers.append(OpenAIProvider())
    if settings.gemini_api_key:
        providers.append(GeminiProvider())

    if not providers:
        raise RuntimeError(
            "No LLM providers configured. Set at least one of ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, GEMINI_API_KEY."
        )
    return providers


def pick_generator_verifier_pairs(
    providers: list[Provider], count: int
) -> list[tuple[Provider, Provider]]:
    """Round-robins a generator across `count` candidates, and pairs each with a
    *different* provider as verifier whenever more than one provider is enabled, so
    a candidate is never fact-checked by the same model/vendor that wrote it."""
    pairs: list[tuple[Provider, Provider]] = []
    for i in range(count):
        generator = providers[i % len(providers)]
        if len(providers) > 1:
            verifier = providers[(i + 1) % len(providers)]
        else:
            verifier = generator
        pairs.append((generator, verifier))
    return pairs
