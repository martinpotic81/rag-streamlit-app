
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        seed=50
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a smart assistant who deeply understands content of the documents provided.\n\n"
         "You are provided with documents which contain private company data.\n\n"
         "Your main function is to understand documents content and answer to user's questions.\n\n"
         "Use emoji where appropriate.\n\n"
         "Summarize long text.\n\n"
         "Use bullets for better presentation.\n\n"
         "Use bold, italic and underlined font to mark important facts.\n\n"
         "If you do not find information regarding the user's request say so politely.\n\n"
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