
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import CHROMA_DIR

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def load_db():
    embeddings = OpenAIEmbeddings(
        #model="text-embedding-3-small"
        model="text-embedding-3-large"
    )

    return Chroma(
        persist_directory="./data/chroma_db",
        embedding_function=embeddings
    )