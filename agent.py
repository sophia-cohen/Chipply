import os
import json
from datetime import datetime, timezone
import anthropic
from exa_py import Exa
from db import get_schema, run_query, _get_secret
from openalex import search_authors, get_author, get_author_works, search_works, search_topics

client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
exa = Exa(_get_secret("EXA_API_KEY"))
MODEL = "claude-sonnet-4-6"
MAX_TOOL_TURNS = 10

SYSTEM_PROMPT = """You are Chipply, an expert data analyst and research agent for Engine Ventures.

Today's date is {today}.

# About Engine Ventures
Engine Ventures is a venture capital firm that invests in very early stage startups. The team focuses on identifying professors and technology with commercial potential — spotting them early before they become widely known. The firm tracks principal investigators (PIs), startups, institutions, and research that could lead to investable opportunities.

# About this database
The database is Engine Ventures' deal flow and relationship management system. It contains:
- **PIs / Professors** — researchers whose work has commercial potential. Professors are ranked with a priority scale: 1 = highest priority, 3 = lowest priority.
- **Startups** — early stage companies, often university spinouts
- **Institutions** — universities and research organizations
- **Deal flow tracking** — how entities move through Engine's pipeline from initial discovery to investment decision
- **Affinity integration** — the main database has an Affinity table that maps entities to their Affinity CRM IDs. The separate Affinity database holds the full interaction history (emails, meetings, notes) with these people and organizations. IMPORTANT: When linking to Affinity profiles, ALWAYS use exactly this URL format: https://engine.affinity.co/persons/{{ID}} — where {{ID}} is the numeric Affinity ID from the database. Do NOT use app.affinity.co or any other domain. Include these links when referencing specific people.

When answering questions, think like a VC analyst: focus on pipeline health, relationship strength, timing, and actionable insights.

{schema}

# Instructions

## Database queries
- You have access to TWO databases via the `run_sql` tool:
  - `main` — the primary database with PIs, startups, institutions, deal flow, and the Affinity ID mapping table.
  - `affinity` — the Affinity interactions database with communication and meeting history, linked by Affinity IDs.
- Set the `db` parameter to "main" or "affinity" to target the correct database.
- To answer questions that span both databases, run separate queries against each and combine the results in your analysis.
- NEVER run INSERT, UPDATE, DELETE, DROP, ALTER, or any mutating statement.
- Analyze the query results thoroughly and provide clear, insightful answers.
- When relevant, highlight trends, anomalies, or interesting patterns in the data.
- If a question is ambiguous, explain your interpretation and the assumptions behind your query.
- For large result sets, summarize the key findings rather than dumping raw data.
- If asked about the schema or database structure, answer from your knowledge of the schema above.
- Do NOT include SQL in your text response — the UI will display it separately.

## Web research
- Use `web_search` to find external context when it would enrich your analysis — industry benchmarks, market data, definitions, news about entities in the database, etc.
- Use `get_page_contents` to read the full text of specific URLs when search snippets aren't enough.
- You can combine database queries with web research in a single answer. For example: query the DB for revenue figures, then search the web for industry comparisons.
- When presenting web research alongside DB results, clearly distinguish between internal data and external sources.
- Cite your sources when using web research.

## OpenAlex (academic research data)
- Use OpenAlex tools to look up professors, their publications, citation metrics, h-index, research topics, and trends.
- `openalex_search_authors` — find researchers by name, returns metrics and institution
- `openalex_get_author` — get detailed profile for a specific researcher (by OpenAlex ID or name)
- `openalex_get_author_works` — get a researcher's top or recent papers
- `openalex_search_works` — search for papers by topic/keyword
- `openalex_search_topics` — explore research fields and trending topics
- Combine OpenAlex data with the internal database to enrich professor profiles — e.g., look up a PI from the database on OpenAlex to get their publication record and citation impact.
- When discovering new PIs, search OpenAlex for prolific researchers in relevant fields and check if they're already in Engine's database.

"""

TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute a read-only SQL SELECT query against either the main database or the Affinity interactions database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute. Must be read-only."
                },
                "db": {
                    "type": "string",
                    "enum": ["main", "affinity"],
                    "description": "Which database to query. 'main' for entity data, 'affinity' for interaction history.",
                    "default": "main"
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web using Exa for external context — industry benchmarks, market data, company info, definitions, news, research papers, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and descriptive for best results."
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_page_contents",
        "description": "Fetch the full text content of a specific URL. Use this when search snippets aren't sufficient and you need to read the full page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from."
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "openalex_search_authors",
        "description": "Search OpenAlex for researchers/professors by name. Returns metrics like h-index, citation count, institution, and research topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Author name or search query."
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of results (default 10, max 50).",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "openalex_get_author",
        "description": "Get detailed profile for a specific researcher — full metrics, institution history, top topics, and publication trends by year.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "OpenAlex author ID (e.g. 'A1234567890') or author name to look up."
                }
            },
            "required": ["author_id"]
        }
    },
    {
        "name": "openalex_get_author_works",
        "description": "Get a researcher's publications, sorted by citation count or recency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "OpenAlex author ID or author name."
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of works to return (default 10).",
                    "default": 10
                },
                "sort": {
                    "type": "string",
                    "enum": ["cited_by_count:desc", "publication_year:desc"],
                    "description": "Sort order. Default is most cited.",
                    "default": "cited_by_count:desc"
                }
            },
            "required": ["author_id"]
        }
    },
    {
        "name": "openalex_search_works",
        "description": "Search for academic papers/research by topic or keyword. Use to find cutting-edge research in a field.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for papers (e.g. 'CRISPR gene editing therapeutics')."
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of results (default 10).",
                    "default": 10
                },
                "filter": {
                    "type": "string",
                    "description": "Optional OpenAlex filter string (e.g. 'publication_year:2024' or 'from_publication_date:2024-01-01')."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "openalex_search_topics",
        "description": "Search for research topics/fields to understand the landscape. Returns topic descriptions and work counts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic search query (e.g. 'synthetic biology', 'quantum computing')."
                },
                "per_page": {
                    "type": "integer",
                    "description": "Number of results (default 10).",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
]


def _handle_run_sql(tool_use) -> dict:
    sql = tool_use.input.get("sql", "")
    db = tool_use.input.get("db", "main")
    try:
        rows, col_names = run_query(sql, db=db)
        result_str = json.dumps(
            {"columns": col_names, "rows": rows, "row_count": len(rows)},
            default=str,
        )
        if len(result_str) > 50000:
            result_str = result_str[:50000] + "\n... (truncated)"
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": result_str}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": f"SQL Error: {e}", "is_error": True}


def _handle_web_search(tool_use) -> dict:
    query = tool_use.input.get("query", "")
    num_results = min(tool_use.input.get("num_results", 5), 10)
    try:
        results = exa.search_and_contents(
            query,
            num_results=num_results,
            text={"max_characters": 3000},
            type="auto",
        )
        formatted = []
        for r in results.results:
            formatted.append({
                "title": r.title,
                "url": r.url,
                "text": r.text[:3000] if r.text else "",
            })
        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(formatted, default=str),
        }
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": f"Search Error: {e}", "is_error": True}


def _handle_get_page_contents(tool_use) -> dict:
    url = tool_use.input.get("url", "")
    try:
        results = exa.get_contents([url], text={"max_characters": 10000})
        if results.results:
            text = results.results[0].text or ""
            return {"type": "tool_result", "tool_use_id": tool_use.id, "content": text[:10000]}
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": "No content found."}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": f"Fetch Error: {e}", "is_error": True}


def _handle_openalex(tool_use) -> dict:
    """Generic handler for all OpenAlex tools."""
    name = tool_use.name
    try:
        if name == "openalex_search_authors":
            result = search_authors(tool_use.input["query"], tool_use.input.get("per_page", 10))
        elif name == "openalex_get_author":
            result = get_author(tool_use.input["author_id"])
        elif name == "openalex_get_author_works":
            result = get_author_works(
                tool_use.input["author_id"],
                tool_use.input.get("per_page", 10),
                tool_use.input.get("sort", "cited_by_count:desc"),
            )
        elif name == "openalex_search_works":
            result = search_works(
                tool_use.input["query"],
                tool_use.input.get("per_page", 10),
                tool_use.input.get("filter"),
            )
        elif name == "openalex_search_topics":
            result = search_topics(tool_use.input["query"], tool_use.input.get("per_page", 10))
        else:
            return {"type": "tool_result", "tool_use_id": tool_use.id, "content": f"Unknown tool: {name}", "is_error": True}

        result_str = json.dumps(result, default=str)
        if len(result_str) > 50000:
            result_str = result_str[:50000] + "\n... (truncated)"
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": result_str}
    except Exception as e:
        return {"type": "tool_result", "tool_use_id": tool_use.id, "content": f"OpenAlex Error: {e}", "is_error": True}


TOOL_HANDLERS = {
    "run_sql": _handle_run_sql,
    "web_search": _handle_web_search,
    "get_page_contents": _handle_get_page_contents,
    "openalex_search_authors": _handle_openalex,
    "openalex_get_author": _handle_openalex,
    "openalex_get_author_works": _handle_openalex,
    "openalex_search_works": _handle_openalex,
    "openalex_search_topics": _handle_openalex,
}


def create_agent(schema: str):
    """Create a stateful agent with the given schema context."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = SYSTEM_PROMPT.format(schema=schema, today=today)
    messages = []

    def chat(user_message: str, status_callback=None) -> dict:
        """Run the agent and return structured result.

        Returns:
            {
                "text": str,           # final text response
                "tool_calls": [        # list of tool calls made
                    {"tool": str, "input": dict, "result_preview": str, "error": bool},
                    ...
                ]
            }
        """
        messages.append({"role": "user", "content": user_message})
        all_tool_calls = []
        turns = 0

        while turns < MAX_TOOL_TURNS:
            turns += 1

            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=system,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as e:
                return {"text": f"API Error: {e}", "tool_calls": all_tool_calls}

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                text_parts = [b.text for b in assistant_content if b.type == "text"]
                return {"text": "\n".join(text_parts), "tool_calls": all_tool_calls}

            tool_results = []
            for tool_use in tool_uses:
                # Notify UI of what's happening
                if status_callback:
                    if tool_use.name == "run_sql":
                        db = tool_use.input.get("db", "main")
                        status_callback(f"Querying {db} database...")
                    elif tool_use.name == "web_search":
                        status_callback(f"Searching: {tool_use.input.get('query', '')[:60]}...")
                    elif tool_use.name == "get_page_contents":
                        status_callback(f"Reading page...")
                    elif tool_use.name.startswith("openalex_"):
                        status_callback(f"Querying OpenAlex...")

                handler = TOOL_HANDLERS.get(tool_use.name)
                if handler:
                    result = handler(tool_use)
                else:
                    result = {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Unknown tool: {tool_use.name}",
                        "is_error": True,
                    }

                tool_results.append(result)

                # Track for UI display
                is_error = result.get("is_error", False)
                preview = result.get("content", "")[:200]
                all_tool_calls.append({
                    "tool": tool_use.name,
                    "input": tool_use.input,
                    "result_preview": preview,
                    "error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})

        return {
            "text": "Reached maximum number of tool calls. Here's what I found so far based on the queries above.",
            "tool_calls": all_tool_calls,
        }

    return chat


def log_query(question: str, tool_calls: list[dict], response_text: str):
    """Log a query to the audit trail in the main database."""
    from db import get_connection
    try:
        conn = get_connection("main")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chipply_query_log (
                id SERIAL PRIMARY KEY,
                asked_at TIMESTAMP DEFAULT NOW(),
                question TEXT NOT NULL,
                tool_calls JSONB,
                response_text TEXT
            )
        """)
        cur.execute(
            "INSERT INTO chipply_query_log (question, tool_calls, response_text) VALUES (%s, %s, %s)",
            (question, json.dumps(tool_calls, default=str), response_text),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass  # Don't break the app if logging fails
