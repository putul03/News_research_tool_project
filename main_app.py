import os
import streamlit as st
from dotenv import load_dotenv

from loader import load_and_split_documents
from embedding_store import create_faiss_index, load_faiss_index
from qa_chain import get_qa_chain

load_dotenv()
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    main_placeholder.text("Loading and splitting documents...")
    docs = load_and_split_documents(urls)
    
    main_placeholder.text("Creating FAISS index and storing embeddings...")
    create_faiss_index(docs, file_path)
    main_placeholder.success("Processing complete!")

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        vectorstore = load_faiss_index(file_path)
        chain = get_qa_chain(vectorstore)
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
