
from langchain_core.documents import Document

def build_documents(chunks, filename, version):
    docs = []
    ids = []

    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_v{version}_{i}"

        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "version": version,
                    "chunk_id": i
                }
            )
        )

        ids.append(doc_id)

    return docs, ids