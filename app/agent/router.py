from typing import Any


class Router:
    """Determine whether the user input and available modalities indicate vision, audio, RAG, or web search workflow paths."""

    def route(self, state: dict[str, Any]) -> list[str]:
        routes = []
        if state.get('image_paths'):
            routes.append('vision')
        if state.get('audio_paths'):
            routes.append('audio')
        if state.get('document_paths') or state.get('user_query'):
            routes.append('rag')
        if state.get('user_query') and 'current' in (state.get('user_query') or '').lower():
            routes.append('web')
        return routes or ['rag']
