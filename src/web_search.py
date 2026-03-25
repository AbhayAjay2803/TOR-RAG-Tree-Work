"""Web search module using DuckDuckGo (free, no API key)."""
from duckduckgo_search import DDGS

def web_search(query: str, max_results: int = 3) -> str:
    """Return a string of snippets from web search results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        snippets = []
        for r in results:
            if r.get('body'):
                snippets.append(r['body'])
        return "\n\n".join(snippets)
    except Exception as e:
        print(f"Web search error: {e}")
        return ""