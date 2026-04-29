
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)
    return splitter.split_text(text)