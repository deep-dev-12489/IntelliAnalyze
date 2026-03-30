import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_PATH = "faiss_index"

# ── 1. Initialize Embeddings ──────────────────────────────────────────────────
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ── 2. Load and Chunk Documents ───────────────────────────────────────────────
def load_and_chunk_pdf(file_path: str):
    """Loads a single PDF and splits it into chunks."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            print(f"Warning: No content found in {file_path}")
            return []
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        return chunks
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# ── 3. Initialize/Load Vector Store ──────────────────────────────────────────
def get_vector_store():
    """Initializes a new vector store or loads an existing one."""
    embeddings = get_embeddings()
    
    if os.path.exists(INDEX_PATH):
        try:
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading index: {e}. Starting fresh.")
    
    return None

def ingest_knowledge_base():
    """Ingests all PDFs from the knowledge_base folder (Initial Setup)."""
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        print(f"Created folder: {KNOWLEDGE_BASE_DIR}")
        return None

    pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF documents found in knowledge_base/.")
        return None

    all_chunks = []
    for pdf in pdf_files:
        path = os.path.join(KNOWLEDGE_BASE_DIR, pdf)
        all_chunks.extend(load_and_chunk_pdf(path))

    if not all_chunks:
        return None

    vectorstore = FAISS.from_documents(all_chunks, get_embeddings())
    vectorstore.save_local(INDEX_PATH)
    print(f"Knowledge base indexed and saved to {INDEX_PATH}")
    return vectorstore

# ── 4. Dynamic Indexing ───────────────────────────────────────────────────────
def update_index(file_path: str):
    """Adds a single new file to the existing FAISS index."""
    if not file_path.endswith('.pdf'):
        print("Only PDF files are supported for dynamic indexing currently.")
        return False
        
    chunks = load_and_chunk_pdf(file_path)
    if not chunks:
        return False
        
    embeddings = get_embeddings()
    vectorstore = get_vector_store()
    
    if vectorstore:
        vectorstore.add_documents(chunks)
        vectorstore.save_local(INDEX_PATH)
        print(f"Updated index with file: {file_path}")
    else:
        # Create fresh index if none exists
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        print(f"Created new index with file: {file_path}")
        
    return True

# ── 5. Retriever Interface ───────────────────────────────────────────────────
def get_retriever():
    """Returns a retriever for the orchestrator."""
    vectorstore = get_vector_store()
    if not vectorstore:
        # Try to ingest from folder if empty
        vectorstore = ingest_knowledge_base()
        
    if vectorstore:
        return vectorstore.as_retriever()
    return None

if __name__ == "__main__":
    # Test ingestion
    ingest_knowledge_base()
