
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "lambda_mult": 0.5}
    )