
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
import time
import re

from vectorstore.chroma_store import load_db
from ingestion.loader import load_file
from ingestion.splitter import split_text
from utils.build_docs import build_documents

# ---------------- LOAD DB ---------------- #
db = load_db()

st.set_page_config(page_title="HSE Assistant", layout="centered")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📂 Document Manager")

mode = st.sidebar.radio("Navigation", ["💬 Chat", "🔍 Search", "📊 Documents"])

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True
)

# Upload logic
if uploaded_files:
    for file in uploaded_files:
        text = load_file(file)
        chunks = split_text(text)

        docs, ids = build_documents(chunks, file.name, version=1)
        db.add_documents(docs, ids=ids)

    st.sidebar.success("✅ Files uploaded!")

# ---------------- FILE EXPLORER ---------------- #
st.sidebar.subheader("📁 Files")

data = db.get()
files = {}

for meta in data["metadatas"]:
    name = meta["source"]
    version = meta["version"]

    if name not in files:
        files[name] = set()
    files[name].add(version)

selected_file = None

for file, versions in files.items():
    if st.sidebar.button(f"📄 {file} (v{max(versions)})"):
        selected_file = file

# Delete
st.sidebar.subheader("🗑 Delete File")
file_to_delete = st.sidebar.text_input("File name")

if st.sidebar.button("Delete"):
    db.delete(where={"source": file_to_delete})
    st.sidebar.success("Deleted!")

# ---------------- MAIN ---------------- #
st.title("🤖 Smart HSE Assistant")

# ---------------- 💬 CHAT ---------------- #
if mode == "💬 Chat":

    st.subheader("💬 Chat with your data")

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (fixed bottom)
    question = st.chat_input("Ask something...")

    if question:
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # -------- RETRIEVAL -------- #
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "filter": {"source": selected_file} if selected_file else None
            }
        )

        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # -------- LLM -------- #
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            streaming=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering user questions using ONLY the provided context.\n\n"
             "Rules:\n"
             "- If answer is not in context, say 'I don't know'\n"
             "- Be concise but include key facts\n"
             "- Use bullet points if helpful\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        chain = prompt | llm

        # -------- STREAMING WITH TYPING EFFECT -------- #
        with st.chat_message("assistant"):
            placeholder = st.empty()

            full_response = ""
            buffer = ""

            time.sleep(0.2)  # thinking delay

            for chunk in chain.stream({
                "context": context,
                "question": question
            }):
                if chunk.content:
                    buffer += chunk.content

                    # Split into sentences
                    sentences = re.split(r'(?<=[.!?])\s+', buffer)

                    complete_sentences = sentences[:-1]
                    buffer = sentences[-1]

                    for sentence in complete_sentences:
                        full_response += sentence + " "

                        placeholder.markdown(full_response + "▌")

                        # adaptive typing speed
                        delay = 0.02 if len(sentence) < 80 else 0.04
                        time.sleep(delay)

            # Flush remaining text
            full_response += buffer
            placeholder.markdown(full_response)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })

# ---------------- 🔍 SEARCH ---------------- #
elif mode == "🔍 Search":

    st.subheader("🔍 Search documents")

    search_query = st.text_input("Search query")

    if search_query:
        retriever = db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(search_query)

        for d in docs:
            st.write(f"📄 {d.metadata['source']} (v{d.metadata['version']})")
            st.write(d.page_content[:300])
            st.divider()

# ---------------- 📊 DOCUMENTS ---------------- #
elif mode == "📊 Documents":

    st.subheader("📊 Indexed Documents")

    if not files:
        st.write("No documents yet 📂")

    for name, versions in files.items():
        st.write(f"📄 **{name}**")
        st.write(f"Versions: {sorted(list(versions))}")
        st.divider()
