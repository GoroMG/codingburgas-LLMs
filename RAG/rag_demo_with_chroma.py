from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load and chunk PDFs, embed, and store in Chroma database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

st.title("RAG Demo: PDF Question-Answering")


@st.cache_resource
def setup_chroma_db(data_dir, db_dir):
    """Load PDFs, chunk, embed, and store in Chroma vector database."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Load and split PDFs
    loaders = [PyPDFLoader(str(pdf)) for pdf in Path(data_dir).glob("*.pdf")]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed and store in Chroma DB
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
    vector_store.persist()
    return vector_store


@st.cache_resource
def load_chroma_db(db_dir):
    """Load Chroma database."""
    return Chroma(persist_directory=db_dir, embedding_function=OpenAIEmbeddings())


if not os.path.exists(DB_DIR):
    st.write("Setting up Chroma database...")
    vector_store = setup_chroma_db(DATA_DIR, DB_DIR)
else:
    vector_store = load_chroma_db(DB_DIR)

# Step 2: Create Retrieval-based QA Chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Step 3: Streamlit UI for Q&A
st.header("Ask a question about the documents")
user_question = st.text_input("Your Question:")

if user_question:
    with st.spinner("Retrieving and generating response..."):
        response = qa_chain.run(user_question)
    st.write("### Response:")
    st.write(response)