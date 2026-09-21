import random

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


def pick_generator_and_verifier(
    providers: list[Provider], tried_generators: set[str]
) -> tuple[Provider, Provider]:
    """Randomly picks a generator (preferring one not already tried for this
    question, so a retry after a failure actually lands on a different provider) and
    a verifier that is never the same provider as the generator, whenever more than
    one is enabled — a candidate should never fact-check its own work."""
    untried = [p for p in providers if p.name not in tried_generators]
    generator = random.choice(untried or providers)

    other_providers = [p for p in providers if p.name != generator.name]
    verifier = random.choice(other_providers) if other_providers else generator

    return generator, verifier
