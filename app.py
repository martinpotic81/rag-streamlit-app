
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import os
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
from vectorstore.chroma_store import load_db
from ingestion.loader import load_file
from ingestion.splitter import split_text
from utils.build_docs import build_documents

# Load DB
db = load_db()

st.set_page_config(page_title="HSE Assistant", layout="centered")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📂 Document Manager")

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

#    db.persist()
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
#    db.persist()
    st.sidebar.success("Deleted!")

# ---------------- MAIN AREA ---------------- #

st.title("🤖 Smart HSE Assistant")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Search", "📊 Documents"])

# ---------------- 💬 CHAT TAB ---------------- #
with tab1:
    st.subheader("💬 Search your data")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display history
    for role, msg in st.session_state.messages:
        st.write(f"{'👤' if role=='user' else '🤖'} {msg}")

    question = st.text_input("Ask me anything from domain data...")

    if question:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "filter": {"source": selected_file} if selected_file else None
            }
        )

        docs = retriever.invoke(question)

        context = "\n\n".join(d.page_content for d in docs)

        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            seed=365
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering user questions using base knowledge provided in stored documents.\n\n"
             "Guidelines:\n"
             "1. Provide very summarized answers using the context below.\n"
             "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
             "3. Use emoji and other helpful marks to point some facts.\n"
             "4. Only use information from the provided context - do not use outside knowledge.\n"
             "5. Summarize long information, ideally in bullets where needed\n"
             "6. If the information is not in the context, say so politely.\n\n"
             "Context:\n{context}"),
            ("human", "{question}")
        ])

        chain = prompt | llm

        response = chain.invoke({
            "context": context,
            "question": question
        }).content

        st.session_state.messages.append(("user", question))
        st.session_state.messages.append(("bot", response))

        st.write("🤖", response)

# ---------------- 🔍 SEARCH TAB ---------------- #
with tab2:
    st.subheader("🔍 Search documents")

    search_query = st.text_input("Search query")

    if search_query:
        retriever = db.as_retriever(search_kwargs={"k": 5})

        docs = retriever.invoke(search_query)

        for d in docs:
            st.write(f"📄 {d.metadata['source']} (v{d.metadata['version']})")
            st.write(d.page_content[:300])
            st.divider()

# ---------------- 📊 DOCUMENTS TAB ---------------- #
with tab3:
    st.subheader("📊 Indexed Documents")

    if not files:
        st.write("No documents yet 📂")

    for name, versions in files.items():
        st.write(f"📄 **{name}**")
        st.write(f"Versions: {sorted(list(versions))}")
        st.divider()