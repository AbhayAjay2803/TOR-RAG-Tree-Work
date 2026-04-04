"""Web search module using DuckDuckGo with proper user-agent, timeout, and retry."""
import requests
import time
from typing import Optional

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def _duckduckgo_search(query: str, max_results: int, retries: int = 2) -> str:
    """Search DuckDuckGo using the HTML endpoint (no API key)."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries):
        try:
            resp = requests.post(url, data=params, headers=headers, timeout=10)
            if resp.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return ""
            # Parse HTML to extract snippets (simplified)
            import re
            snippets = re.findall(r'<a class="result__a" href="[^"]*">([^<]+)</a>', resp.text)
            if not snippets:
                snippets = re.findall(r'<div class="result__snippet">([^<]+)</div>', resp.text)
            results = []
            for snippet in snippets[:max_results]:
                clean = re.sub(r'<[^>]+>', '', snippet).strip()
                if clean:
                    results.append(clean)
            return "\n\n".join(results)
        except Exception as e:
            print(f"DuckDuckGo search error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return ""

def web_search(query: str, max_results: int = 3, backend: str = "duckduckgo") -> str:
    """Perform a web search using the specified backend."""
    if backend == "duckduckgo":
        return _duckduckgo_search(query, max_results)
    elif backend == "wikipedia":
        from .wiki_search import wikipedia_search
        return wikipedia_search(query, max_results)
    else:
        print(f"Unknown backend: {backend}, falling back to DuckDuckGo")
        return _duckduckgo_search(query, max_results)