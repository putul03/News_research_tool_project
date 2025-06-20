#For Embedding and Storing in FAISS
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def create_faiss_index(docs, file_path):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

def load_faiss_index(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
