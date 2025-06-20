#For Retrieval and Answer Generation
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

def get_qa_chain(vectorstore, temperature=0.9, max_tokens=500):
    llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
    retriever = vectorstore.as_retriever()
    return RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
