import requests

BASE_URL = "https://api.openalex.org"
HEADERS = {"User-Agent": "mailto:team@engineventures.com"}


def search_authors(query: str, per_page: int = 10) -> dict:
    """Search for authors/researchers by name."""
    resp = requests.get(
        f"{BASE_URL}/authors",
        params={"search": query, "per_page": per_page},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for a in data.get("results", []):
        results.append({
            "openalex_id": a.get("id", ""),
            "name": a.get("display_name", ""),
            "works_count": a.get("works_count", 0),
            "cited_by_count": a.get("cited_by_count", 0),
            "h_index": a.get("summary_stats", {}).get("h_index", 0),
            "i10_index": a.get("summary_stats", {}).get("i10_index", 0),
            "last_known_institution": (a.get("last_known_institutions") or [{}])[0].get("display_name", "") if a.get("last_known_institutions") else "",
            "topics": [t.get("display_name", "") for t in (a.get("topics") or [])[:5]],
        })
    return {"count": data.get("meta", {}).get("count", 0), "results": results}


def get_author(author_id: str) -> dict:
    """Get detailed info for a specific author by OpenAlex ID or name."""
    # If it looks like an ID, fetch directly; otherwise search
    if author_id.startswith("https://") or author_id.startswith("A"):
        resp = requests.get(f"{BASE_URL}/authors/{author_id}", headers=HEADERS)
    else:
        resp = requests.get(
            f"{BASE_URL}/authors",
            params={"search": author_id, "per_page": 1},
            headers=HEADERS,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {"error": "Author not found"}
        resp = requests.get(f"{BASE_URL}/authors/{results[0]['id']}", headers=HEADERS)

    resp.raise_for_status()
    a = resp.json()
    return {
        "openalex_id": a.get("id", ""),
        "name": a.get("display_name", ""),
        "works_count": a.get("works_count", 0),
        "cited_by_count": a.get("cited_by_count", 0),
        "h_index": a.get("summary_stats", {}).get("h_index", 0),
        "i10_index": a.get("summary_stats", {}).get("i10_index", 0),
        "2yr_mean_citedness": a.get("summary_stats", {}).get("2yr_mean_citedness", 0),
        "last_known_institutions": [
            inst.get("display_name", "") for inst in (a.get("last_known_institutions") or [])
        ],
        "topics": [
            {"name": t.get("display_name", ""), "count": t.get("count", 0)}
            for t in (a.get("topics") or [])[:10]
        ],
        "counts_by_year": (a.get("counts_by_year") or [])[:5],
    }


def get_author_works(author_id: str, per_page: int = 10, sort: str = "cited_by_count:desc") -> dict:
    """Get recent or top works for an author."""
    # Normalize ID
    if not author_id.startswith("https://"):
        if author_id.startswith("A"):
            author_id = f"https://openalex.org/{author_id}"
        else:
            # Search by name first
            search = search_authors(author_id, per_page=1)
            if not search["results"]:
                return {"error": "Author not found"}
            author_id = search["results"][0]["openalex_id"]

    resp = requests.get(
        f"{BASE_URL}/works",
        params={
            "filter": f"author.id:{author_id}",
            "sort": sort,
            "per_page": per_page,
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    works = []
    for w in data.get("results", []):
        works.append({
            "title": w.get("title", ""),
            "publication_year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "type": w.get("type", ""),
            "doi": w.get("doi", ""),
            "is_open_access": w.get("open_access", {}).get("is_oa", False),
            "source": (w.get("primary_location") or {}).get("source", {}).get("display_name", "") if w.get("primary_location") else "",
        })
    return {"total_works": data.get("meta", {}).get("count", 0), "works": works}


def search_works(query: str, per_page: int = 10, filter_str: str = None) -> dict:
    """Search for research works/papers. Optional filter for advanced queries."""
    params = {"search": query, "per_page": per_page, "sort": "relevance_score:desc"}
    if filter_str:
        params["filter"] = filter_str
    resp = requests.get(f"{BASE_URL}/works", params=params, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    works = []
    for w in data.get("results", []):
        authors = [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]]
        works.append({
            "title": w.get("title", ""),
            "authors": authors,
            "publication_year": w.get("publication_year"),
            "cited_by_count": w.get("cited_by_count", 0),
            "doi": w.get("doi", ""),
            "type": w.get("type", ""),
            "topics": [c.get("display_name", "") for c in (w.get("topics") or [])[:3]],
        })
    return {"count": data.get("meta", {}).get("count", 0), "works": works}


def search_topics(query: str, per_page: int = 10) -> dict:
    """Search for research topics/concepts."""
    resp = requests.get(
        f"{BASE_URL}/topics",
        params={"search": query, "per_page": per_page},
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    topics = []
    for t in data.get("results", []):
        topics.append({
            "openalex_id": t.get("id", ""),
            "name": t.get("display_name", ""),
            "works_count": t.get("works_count", 0),
            "cited_by_count": t.get("cited_by_count", 0),
            "description": t.get("description", ""),
        })
    return {"count": data.get("meta", {}).get("count", 0), "topics": topics}
