📚 RAG Demo: PDF Question-Answering with Chroma and OpenAI

This project demonstrates a Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents. By combining Chroma for vector storage and OpenAI's embeddings, users can query document data efficiently and get accurate responses using a language model.
🚀 Features

    Load and process PDF files into a vector database (Chroma).
    Automatically detect new PDFs and update the database.
    Use chunking and embeddings to create a searchable document index.
    Retrieve the most relevant document sections for user queries.
    Generate answers with OpenAI’s gpt-3.5-turbo model.
    User-friendly Streamlit interface for asking questions.

🛠️ Setup Instructions
1. Clone the Repository

git clone https://github.com/GoroMG/codingburgas-LLMs.git
cd rag-pdf-qa-demo

2. Install Dependencies

Create a Python virtual environment and install the required packages:

python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
pip install -r requirements.txt

3. Set Up Environment Variables

Create a .env file in the root directory and add your OpenAI API key:

OPENAI_API_KEY=your_openai_api_key

4. Prepare Data

    Place your PDF files in the data folder (e.g., RAG/data).
    The system will automatically detect and process new PDFs during runtime.

🖥️ Usage
Run the Application

Start the Streamlit app:

streamlit run rag_demo_with_chroma.py

Interact with the App

    Open the app in your browser (e.g., http://localhost:8501).
    Upload your PDF documents into the data folder.
    Enter your question in the text input box.
    View the answer and the retrieved document snippets!

🔍 How It Works
1. Indexing Documents

    PDFs are loaded using PyPDFLoader and split into smaller chunks using RecursiveCharacterTextSplitter.
    Each chunk is embedded into a vector space using OpenAI's text-embedding-ada-002.
    The embeddings are stored in a Chroma database for efficient retrieval.

2. Retrieval

    User queries are processed using similarity search with Chroma.
    Top 10 relevant chunks are retrieved and scored.
    The top 5 chunks with the highest scores are selected for further processing.

3. Augmentation

    The content of the top 5 chunks is combined into a single context string.
    This context is fed into the language model for generating answers.

4. Generation

    OpenAI’s gpt-3.5-turbo generates a response based on the query and context.

📂 Project Structure

'''
├── data/              # Folder for PDF files
├── chroma_db/         # Persistent Chroma database
├── .env               # Environment variables
├── rag_demo_with_chroma.py  # Main application script
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
'''

🛡️ Error Handling

    Missing API Key: The app will stop with an error if the OpenAI API key is not provided.
    No PDFs Found: If no PDFs are in the data folder, the app will prompt you to add files.
    New PDFs: The app automatically detects and processes new PDFs when added to the data folder.
    General Exceptions: Errors during processing are logged and a fallback response is provided.

🔧 Customization
Modify Chunk Size or Overlap

You can adjust chunking parameters in the setup_chroma_db and load_chroma_db functions:

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

Change Number of Retrieved Chunks

To retrieve a different number of documents for processing, update the k parameter:

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

📋 Requirements

    Python 3.10 or higher
    OpenAI API key

📜 License

This project is licensed under the MIT License.
💡 Inspiration

This project showcases how RAG systems can improve question-answering by combining structured retrieval with generative models.
