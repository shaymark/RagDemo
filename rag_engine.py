"""
rag_engine.py — Core RAG functions
====================================
This is the single source of truth for all RAG pipeline logic.
Import from here in app.py and any other scripts.

Functions:
  load_documents     — load .txt files from a directory
  chunk_text         — split text into overlapping chunks
  prepare_chunks     — convert documents to ChromaDB format
  build_vector_store — embed chunks and store in ChromaDB
  load_collection    — load an existing ChromaDB collection
  retrieve           — find top-k most relevant chunks for a query
"""

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "aquabot_docs"


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def load_documents(docs_dir: str) -> list[dict]:
    """
    Load all .txt files from a directory.
    Returns a list of dicts: {id, text, source}
    """
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r") as f:
            text = f.read()
        documents.append({
            "id": filename.replace(".txt", ""),
            "text": text,
            "source": filename,
        })
        print(f"  Loaded: {filename} ({len(text)} chars)")
    return documents


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split a long text into overlapping chunks.

    chunk_size: max characters per chunk
    overlap:    how many characters the next chunk shares with the previous
                (prevents relevant sentences being cut at a boundary)
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def prepare_chunks(documents: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Convert documents into (ids, texts, metadatas) ready for ChromaDB.
    """
    ids, texts, metadatas = [], [], []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}__chunk{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})
    print(f"\n  Total chunks created: {len(ids)}")
    return ids, texts, metadatas


# ─────────────────────────────────────────────
# STORE
# ─────────────────────────────────────────────

def build_vector_store(
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Embed all chunks and store them in ChromaDB.

    Uses the all-MiniLM-L6-v2 sentence-transformer model (384-dim vectors).
    Always deletes and recreates the collection to ensure a clean state.
    Persists to disk at chroma_path so data survives restarts.
    """
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection to start fresh
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n  Embedding and storing {len(ids)} chunks...")
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"  Done! '{collection_name}' has {collection.count()} vectors.")
    return collection


# ─────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────

def load_collection(
    chroma_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk.
    Raises RuntimeError if the collection doesn't exist yet.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.DefaultEmbeddingFunction()
    existing = [c.name for c in client.list_collections()]

    if collection_name not in existing:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in '{chroma_path}'. "
            "Run build_vector_store() first."
        )

    collection = client.get_collection(name=collection_name, embedding_function=ef)
    return collection


def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Convert query to a vector and return the top_k most similar chunks.

    Returns a list of dicts: {text, source, similarity}
    similarity is in range [0, 1] — higher is more relevant.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "similarity": 1 - results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
