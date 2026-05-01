# app.py

import streamlit as st
from rag import build_rag_chain, query

st.set_page_config(page_title="Local RAG", page_icon="📄")

st.title("📄 Chat with your documents")
st.caption("Fully local — no API keys, no data leaves your machine")

# Load chain once
@st.cache_resource
def load_chain():
    return build_rag_chain()

chain = load_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = query(chain, question)

        st.write(answer)

        # Show sources
        if sources:
            with st.expander("Sources", expanded=False):
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get("source", "unknown")
                    preview = doc.page_content[:200].replace("\n", " ")

                    st.markdown(f"**Source {i}** — `{source}`")
                    st.markdown(f"> {preview}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})