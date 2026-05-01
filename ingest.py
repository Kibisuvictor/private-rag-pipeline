# ingest.py

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os
import shutil

DOCS_PATH = "./data"
CHROMA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"  # ✅ match rag.py


def load_documents():
    loader = DirectoryLoader(
        DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # ✅ better for context
        chunk_overlap=150,   # ✅ smoother transitions
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks):
    # ✅ Reset DB to avoid duplicates
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    # ✅ Ensure persistence
    db.persist()

    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_PATH}")
    return db


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    embed_and_store(chunks)