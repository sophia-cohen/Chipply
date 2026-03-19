import streamlit as st
from db import get_schema
from agent import create_agent, log_query
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chipply", page_icon="logo.png", layout="wide")
st.image("logo.png", width=200)


# Cache schema so it doesn't re-introspect on every session
@st.cache_data(ttl=3600, show_spinner=False)
def load_schema():
    return get_schema()


# Initialize agent (cached per session)
if "agent" not in st.session_state:
    with st.spinner("Connecting to database and reading schema..."):
        try:
            schema = load_schema()
            st.session_state.schema = schema
            st.session_state.agent = create_agent(schema)
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            st.info("Make sure your `.env` file has valid `DATABASE_URL` and `ANTHROPIC_API_KEY` values.")
            st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# New conversation button
if st.session_state.messages:
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.agent = create_agent(st.session_state.schema)
        st.rerun()


def render_tool_calls(tool_calls):
    """Render tool calls as expandable sections."""
    for tc in tool_calls:
        if tc["tool"] == "run_sql":
            db = tc["input"].get("db", "main")
            sql = tc["input"].get("sql", "")
            label = f"SQL query ({db} database)"
            if tc["error"]:
                label += " — error"
            with st.expander(label, expanded=False):
                st.code(sql, language="sql")
                if tc["error"]:
                    st.error(tc["result_preview"])
        elif tc["tool"] == "web_search":
            query = tc["input"].get("query", "")
            with st.expander(f"Web search: {query[:80]}", expanded=False):
                st.write(tc["result_preview"][:500])
        elif tc["tool"] == "get_page_contents":
            url = tc["input"].get("url", "")
            with st.expander(f"Read page: {url[:80]}", expanded=False):
                st.write(tc["result_preview"][:500])
        elif tc["tool"].startswith("openalex_"):
            tool_label = tc["tool"].replace("openalex_", "OpenAlex: ").replace("_", " ")
            query = tc["input"].get("query", tc["input"].get("author_id", ""))
            with st.expander(f"{tool_label} — {query[:60]}", expanded=False):
                st.write(tc["result_preview"][:500])


def render_message(msg):
    """Render a single message with its tool calls."""
    with st.chat_message(msg["role"]):
        if msg.get("tool_calls"):
            render_tool_calls(msg["tool_calls"])
        st.markdown(msg["content"])


# Render chat history
for msg in st.session_state.messages:
    render_message(msg)

# Chat input
if prompt := st.chat_input("Ask anything about your data..."):
    # Show user message
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response with live status updates
    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)

        def update_status(msg):
            status.update(label=msg)

        try:
            result = st.session_state.agent(prompt, status_callback=update_status)
            text = result["text"]
            tool_calls = result["tool_calls"]
        except Exception as e:
            text = f"Error: {e}"
            tool_calls = []

        status.update(label="Done", state="complete", expanded=False)

        if tool_calls:
            render_tool_calls(tool_calls)
        st.markdown(text)

    assistant_msg = {"role": "assistant", "content": text, "tool_calls": tool_calls}
    st.session_state.messages.append(assistant_msg)

    # Log to audit trail
    log_query(prompt, tool_calls, text)
