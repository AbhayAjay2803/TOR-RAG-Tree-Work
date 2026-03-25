"""Web search module with pluggable backends."""
import requests
from typing import Optional

# ---------- Wikipedia backend ----------
def _wikipedia_search(query: str, max_results: int) -> str:
    try:
        # Search for pages
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results,
            "utf8": 1
        }
        resp = requests.get(search_url, params=params, timeout=5)
        data = resp.json()
        if not data.get("query", {}).get("search"):
            return ""

        # Get page summaries
        page_titles = [item["title"] for item in data["query"]["search"]]
        summaries = []
        for title in page_titles:
            params = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "titles": title,
                "format": "json",
                "utf8": 1
            }
            resp = requests.get(search_url, params=params, timeout=5)
            page_data = resp.json()
            pages = page_data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "extract" in page:
                    summaries.append(page["extract"])
                    break
        return "\n\n".join(summaries[:max_results])
    except Exception as e:
        print(f"Wikipedia search error: {e}")
        return ""

# ---------- Google backend (scraping) ----------
def _google_search(query: str, max_results: int) -> str:
    try:
        from googlesearch import search
        results = []
        for url in search(query, num_results=max_results):
            # For each URL, we could fetch the page, but that's heavy. 
            # Instead, we'll just return titles/snippets? 
            # googlesearch-python returns URLs; we can optionally fetch content.
            # Simpler: return the URLs as a list. 
            # To get snippets, we'd need to parse the HTML. For brevity, just URLs.
            results.append(url)
        if not results:
            return ""
        # Return a list of URLs as plain text (not ideal but works)
        return "\n".join(results)
    except Exception as e:
        print(f"Google search error: {e}")
        return ""

# ---------- DuckDuckGo backend (original) ----------
def _duckduckgo_search(query: str, max_results: int) -> str:
    try:
        from duckduckgo_search import DDGS
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
        print(f"DuckDuckGo search error: {e}")
        return ""

# ---------- Main entry point ----------
def web_search(query: str, max_results: int = 3, backend: str = "wikipedia") -> str:
    """Perform a web search using the specified backend."""
    if backend == "wikipedia":
        return _wikipedia_search(query, max_results)
    elif backend == "google":
        return _google_search(query, max_results)
    elif backend == "duckduckgo":
        return _duckduckgo_search(query, max_results)
    else:
        print(f"Unknown search backend: {backend}. Falling back to Wikipedia.")
        return _wikipedia_search(query, max_results)