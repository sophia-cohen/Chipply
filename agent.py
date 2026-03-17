import os
import json
import anthropic
from exa_py import Exa
from db import get_schema, run_query, _get_secret

client = anthropic.Anthropic(api_key=_get_secret("ANTHROPIC_API_KEY"))
exa = Exa(_get_secret("EXA_API_KEY"))
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert data analyst and research agent connected to a PostgreSQL database (hosted on Neon) and the web via Exa search.
You deeply understand the database schema and can write precise, efficient SQL to answer any question about the data.

{schema}

# Instructions

## Database queries
- When the user asks a question, determine if you need to query the database to answer it.
- If yes, call the `run_sql` tool with a SELECT query. NEVER run INSERT, UPDATE, DELETE, DROP, ALTER, or any mutating statement.
- Analyze the query results thoroughly and provide clear, insightful answers.
- When relevant, highlight trends, anomalies, or interesting patterns in the data.
- If a question is ambiguous, explain your interpretation and the assumptions behind your query.
- For large result sets, summarize the key findings rather than dumping raw data.
- If asked about the schema or database structure, answer from your knowledge of the schema above.
- Always show the SQL you ran so the user can learn and verify.

## Web research
- Use `web_search` to find external context when it would enrich your analysis — industry benchmarks, market data, definitions, news about entities in the database, etc.
- Use `get_page_contents` to read the full text of specific URLs when search snippets aren't enough.
- You can combine database queries with web research in a single answer. For example: query the DB for revenue figures, then search the web for industry comparisons.
- When presenting web research alongside DB results, clearly distinguish between internal data and external sources.
- Cite your sources when using web research.
"""

TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute a read-only SQL SELECT query against the database and return the results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute. Must be read-only."
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
    }
]


def _handle_run_sql(tool_use) -> dict:
    sql = tool_use.input.get("sql", "")
    try:
        rows, col_names = run_query(sql)
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


TOOL_HANDLERS = {
    "run_sql": _handle_run_sql,
    "web_search": _handle_web_search,
    "get_page_contents": _handle_get_page_contents,
}


def create_agent(schema: str):
    """Create a stateful agent with the given schema context."""
    system = SYSTEM_PROMPT.format(schema=schema)
    messages = []

    def chat(user_message: str) -> str:
        messages.append({"role": "user", "content": user_message})

        # Agentic loop: keep going until the model stops calling tools
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if b.type == "tool_use"]
            if not tool_uses:
                text_parts = [b.text for b in assistant_content if b.type == "text"]
                return "\n".join(text_parts)

            tool_results = []
            for tool_use in tool_uses:
                handler = TOOL_HANDLERS.get(tool_use.name)
                if handler:
                    tool_results.append(handler(tool_use))
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": f"Unknown tool: {tool_use.name}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

    return chat
