"""Wikipedia search using the official API (free, no rate limits)."""
import requests

def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Return a string of summaries from Wikipedia search."""
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