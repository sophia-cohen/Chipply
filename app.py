import streamlit as st
from db import get_schema
from agent import create_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chipply", page_icon="logo.png", layout="wide")
st.image("logo.png", width=200)
st.caption("Talk to your database in plain English")

# Initialize schema and agent (cached per session)
if "agent" not in st.session_state:
    with st.spinner("Connecting to database and reading schema..."):
        try:
            schema = get_schema()
            st.session_state.schema = schema
            st.session_state.agent = create_agent(schema)
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            st.info("Make sure your `.env` file has valid `DATABASE_URL` and `ANTHROPIC_API_KEY` values.")
            st.stop()

# Sidebar: show schema
with st.sidebar:
    st.subheader("Database Schema")
    st.markdown(st.session_state.schema, unsafe_allow_html=False)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your data..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = st.session_state.agent(prompt)
            except Exception as e:
                response = f"Error: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
