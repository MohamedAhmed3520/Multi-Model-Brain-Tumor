from app.schemas.web import WebEvidence


class WebSearch:
    """External web search abstraction returning structured web evidence."""

    def __init__(self, provider: str = 'open-web'):
        self.provider = provider

    def search(self, query: str) -> list[WebEvidence]:
        # Placeholder deterministic implementation for a production skeleton.
        return [
            WebEvidence(
                title='No external results available in local skeleton mode',
                url='https://example.invalid/no-results',
                domain='example.invalid',
                snippet='Replace this placeholder with a configured provider such as Tavily, SerpAPI, or OpenRouter-based web search.',
                rank=1,
                relevance_score=0.0,
            )
        ]
