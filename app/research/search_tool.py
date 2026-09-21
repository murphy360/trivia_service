import httpx
from pydantic import BaseModel

from app.core.config import get_settings

TAVILY_URL = "https://api.tavily.com/search"

# Provider-neutral tool description. Each provider adapter translates this into its
# own function/tool-calling schema, but the semantics (and therefore fact-check
# behavior) stay identical across vendors.
TOOL_NAME = "web_search"
TOOL_DESCRIPTION = "Search the web for current, authoritative information to verify a factual claim."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."},
    },
    "required": ["query"],
}


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchTool:
    """Thin wrapper around a single web-search API, shared by every provider's
    fact-check step so verification behavior is comparable across vendors instead
    of depending on whichever native web-search tool that vendor happens to ship."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_settings().search_api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not self._api_key:
            raise RuntimeError(
                "SEARCH_API_KEY is not configured; fact-checking requires a search API key."
            )
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                TAVILY_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                },
            )
            response.raise_for_status()
            data = response.json()

        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in data.get("results", [])
        ]
