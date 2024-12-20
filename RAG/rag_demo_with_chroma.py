import logging
from dotenv import load_dotenv
import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Setup logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables (e.g., API keys)
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    st.error("OPENAI_API_KEY not found in environment variables!")
    st.stop()

# Constants for directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Streamlit app title
st.title("RAG Demo: PDF Question-Answering")

@st.cache_resource
def setup_chroma_db(data_dir, db_dir):
    """
    Setup the Chroma vector database:
    - Load PDFs from the `data_dir`.
    - Chunk the data for efficient retrieval.
    - Embed the chunks into a vector space using OpenAI embeddings.
    - Persist the embeddings in a Chroma database.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Load all PDFs in the directory
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    loaders = [PyPDFLoader(str(pdf)) for pdf in pdf_files]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    logger.info(f"Loaded {len(documents)} documents from {len(pdf_files)} PDFs.")

    # Split documents into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # Embed chunks using OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma(
        collection_name="test_collection",
        embedding_function=embeddings,
        persist_directory=db_dir,
    )

    # Add chunks in batches to the vector store for performance
    max_batch_size = 5000
    for i in range(0, len(chunks), max_batch_size):
        batch = chunks[i:i + max_batch_size]
        vector_store.add_documents(batch)
        logger.info(f"Added batch {i // max_batch_size + 1} of {len(batch)} chunks.")

    logger.info(f"Chroma database setup complete with {len(chunks)} chunks.")
    return vector_store

@st.cache_resource
def load_chroma_db(data_dir, db_dir):
    """
    Load the Chroma vector database. If new PDFs are detected, process them and update the database.
    """
    # Initialize embeddings and Chroma database
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma(
        collection_name="test_collection",
        persist_directory=db_dir,
        embedding_function=embeddings,
    )

    # Retrieve metadata from Chroma to find already indexed files
    try:
        existing_metadata = vector_store.get()["metadatas"]
        existing_sources = {meta["source"] for meta in existing_metadata if "source" in meta}
    except KeyError:
        logger.info("No existing metadata found in the Chroma database.")
        existing_sources = set()

    # Check for new PDFs in the data directory
    new_pdfs = [pdf for pdf in Path(data_dir).glob("*.pdf") if str(pdf) not in existing_sources]

    if new_pdfs:
        logger.info(f"New PDFs detected: {[str(pdf) for pdf in new_pdfs]}")
        loaders = [PyPDFLoader(str(pdf)) for pdf in new_pdfs]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = splitter.split_documents(documents)

        # Add new chunks to the vector store
        vector_store.add_documents(new_chunks)
        logger.info(f"Processed and added {len(new_chunks)} chunks from new PDFs.")

    logger.info("Chroma database loaded.")
    return vector_store


# Initialize the Chroma database
if not os.path.exists(DB_DIR):
    st.write("Setting up Chroma database...")
    vector_store = setup_chroma_db(DATA_DIR, DB_DIR)
else:
    vector_store = load_chroma_db(DATA_DIR, DB_DIR)

# Initialize the retriever and QA chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Streamlit user interface for Q&A
st.header("Ask a question about the documents")
user_question = st.text_input("Your Question:")

if user_question:
    logger.info(f"User query: {user_question}")
    with st.spinner("Retrieving and generating response..."):
        try:
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(user_question, k=10)

            # Sort results by score and select the top 5
            top_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
            top_docs = [result[0] for result in top_results]  # Extract documents

            # Log the selected top results
            logger.info(f"Top 5 results based on score: {[doc.page_content[:100] for doc in top_docs]}")

            # Manually construct the context for the LLM from top_docs
            context = "\n\n".join([doc.page_content for doc in top_docs])

            # Create a query with the context embedded
            qa_input = {"query": user_question, "context": context}

            # Use the QA chain directly with the prebuilt context
            response = qa_chain.invoke(qa_input)
            logger.info(f"Generated response: {response['result']}")
        except Exception as e:
            logger.error(f"Error during retrieval or generation: {str(e)}")
            response = "I'm sorry, but I couldn't retrieve relevant information."

    # Display the response
    st.write("### Response:")
    st.write(response['result'] if isinstance(response, dict) and 'result' in response else response)
