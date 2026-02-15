import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
PDF_PATH = "../Story.pdf"   # Ensure this file exists in the folder
DB_PATH = "./chroma_db"

def main():
    # 1. Clear old database if it exists (to avoid duplicates)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    # 2. Load PDF
    print(f"Loading {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 3. Split Text
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 4. Initialize Embeddings (CPU/MPS friendly)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Create and Save Vector Store
    print("Creating Chroma Database...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print(f"✅ Ingestion Complete! Database saved to {DB_PATH}")

if __name__ == "__main__":
    main()