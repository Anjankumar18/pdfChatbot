# app.py — Streamlit UI
import streamlit as st
import tempfile
import os
from pathlib import Path

# ── Load secrets (works locally + Streamlit Cloud) ────────────
def load_secrets():
    try:
        os.environ["OPENAI_API_KEY"]       = st.secrets["OPENAI_API_KEY"]
        os.environ["LANGCHAIN_API_KEY"]    = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"]    = st.secrets["LANGCHAIN_PROJECT"]
        os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()

load_secrets()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_core import build_chat_graph, ask

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# ── Session State ─────────────────────────────────────────────
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "chat_graph" not in st.session_state:
    st.session_state.chat_graph = None
if "history"    not in st.session_state:
    st.session_state.history    = ChatMessageHistory()
if "num_pages"  not in st.session_state:
    st.session_state.num_pages  = 0
if "file_key"   not in st.session_state:
    st.session_state.file_key   = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or Word files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        file_key = "_".join(sorted([f.name for f in uploaded_files]))

        if st.session_state.file_key != file_key:
            with st.spinner("Processing documents..."):
                pages     = []
                tmp_paths = []

                for file in uploaded_files:
                    suffix = Path(file.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(file.read())
                        tmp_paths.append(tmp.name)

                    loader = (PyPDFLoader(tmp_paths[-1])
                              if suffix == ".pdf"
                              else Docx2txtLoader(tmp_paths[-1]))
                    pages.extend(loader.load())

                for p in tmp_paths:
                    if os.path.exists(p):
                        os.unlink(p)

                if pages:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50
                    )
                    chunks = splitter.split_documents(pages)

                    st.session_state.chat_graph = build_chat_graph(pages, chunks)
                    st.session_state.history    = ChatMessageHistory()
                    st.session_state.messages   = []
                    st.session_state.num_pages  = len(pages)
                    st.session_state.file_key   = file_key

        st.success(f"✅ {len(uploaded_files)} file(s) ready")
        st.caption(f"📊 {st.session_state.num_pages} pages")
        for f in uploaded_files:
            st.caption(f"• {f.name}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history  = ChatMessageHistory()
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages   = []
            st.session_state.history    = ChatMessageHistory()
            st.session_state.chat_graph = None
            st.session_state.file_key   = None
            st.rerun()

    st.divider()
    st.caption("Built with LangChain + LangGraph + Streamlit")

# ── Main Area ─────────────────────────────────────────────────
st.title("📄 PDF Chatbot")
st.caption("Upload documents in the sidebar, then ask questions!")

if not st.session_state.chat_graph:
    st.info("👈 Upload PDF or Word files in the sidebar to get started!")
else:
    st.success("✅ Ready! Ask me anything about your documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input(
    "Ask about your documents...",
    disabled=st.session_state.chat_graph is None
):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({
        "role": "user", "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask(
                    st.session_state.chat_graph,
                    question,
                    st.session_state.history
                )
            except Exception as e:
                answer = f"Something went wrong: {str(e)}"
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant", "content": answer
    })