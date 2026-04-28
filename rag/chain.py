
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant.\n"
         "Use ONLY the provided context.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain