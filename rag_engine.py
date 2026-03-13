"""
rag_engine.py — Core RAG functions (Pinecone backend)
======================================================
All RAG pipeline logic. Import from app.py and any other scripts.

Functions:
  load_documents         — load .txt files from a directory
  chunk_text             — split text into overlapping chunks
  prepare_chunks         — convert documents to (ids, texts, metadatas)
  build_vector_store     — full rebuild: embed all chunks into Pinecone (startup only)
  upsert_document        — incremental: add/replace one file's vectors
  delete_document_vectors— incremental: remove one file's vectors
  get_index              — return the Pinecone Index handle
  retrieve               — find top-k most relevant chunks for a query
"""

import os
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from fastembed import TextEmbedding

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "aquabot-rag")
DIMENSION  = 384      # all-MiniLM-L6-v2 output size
METRIC     = "cosine"

_embedder: TextEmbedding | None = None  # lazy-loaded singleton


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _get_pinecone() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def _ensure_index() -> None:
    """Create the Pinecone index if it doesn't exist yet. Waits until ready."""
    pc = _get_pinecone()
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status.ready:
            time.sleep(1)
        print(f"  Index '{INDEX_NAME}' is ready.")


def get_index():
    """Return the Pinecone index handle. Raises RuntimeError if the index doesn't exist."""
    pc = _get_pinecone()
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        raise RuntimeError(
            f"Pinecone index '{INDEX_NAME}' not found. Run build_vector_store() first."
        )
    return pc.Index(INDEX_NAME)


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
    overlap:    characters shared between consecutive chunks
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
    Convert documents into (ids, texts, metadatas) tuples ready for Pinecone.
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
) -> int:
    """
    Full rebuild: delete ALL vectors then re-embed from scratch.
    Only called on startup when the index is empty. Returns chunk count.
    """
    _ensure_index()
    index = _get_pinecone().Index(INDEX_NAME)
    try:
        index.delete(delete_all=True)
    except Exception:
        pass  # 404 "Namespace not found" is expected on a brand-new empty index

    embedder = _get_embedder()
    embeddings = list(embedder.embed(texts))   # batch embed all chunks at once
    vectors = []
    for chunk_id, text, meta, emb in zip(ids, texts, metadatas, embeddings):
        vectors.append({
            "id": chunk_id,
            "values": emb.tolist(),
            "metadata": {**meta, "text": text},  # text stored here for retrieval
        })

    print(f"\n  Embedding and upserting {len(vectors)} chunks...")
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100])
    print(f"  Done! '{INDEX_NAME}' updated.")
    return len(vectors)


def upsert_document(filename: str, text: str) -> int:
    """
    Add/replace one document's vectors.
    Deletes only that file's old vectors first — other documents are never touched.
    Returns the number of new chunks inserted.
    """
    _ensure_index()
    index = _get_pinecone().Index(INDEX_NAME)

    # Remove only this file's existing vectors
    index.delete(filter={"source": filename})

    doc_id = Path(filename).stem
    ids, texts, metadatas = prepare_chunks(
        [{"id": doc_id, "text": text, "source": filename}]
    )

    embedder = _get_embedder()
    embeddings = list(embedder.embed(texts))   # batch embed
    vectors = []
    for chunk_id, chunk_str, meta, emb in zip(ids, texts, metadatas, embeddings):
        vectors.append({
            "id": chunk_id,
            "values": emb.tolist(),
            "metadata": {**meta, "text": chunk_str},
        })

    if vectors:
        for i in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[i:i + 100])
    return len(vectors)


def delete_document_vectors(filename: str) -> None:
    """
    Delete all vectors belonging to one file. Other files are untouched.
    """
    index = _get_pinecone().Index(INDEX_NAME)
    index.delete(filter={"source": filename})


# ─────────────────────────────────────────────
# RETRIEVE
# ─────────────────────────────────────────────

def retrieve(index, query: str, top_k: int = 3) -> list[dict]:
    """
    Convert query to a vector and return the top_k most similar chunks.

    Returns a list of dicts: {text, source, chunk_index, similarity}
    similarity is the cosine score from Pinecone (higher = more relevant).
    """
    embedder = _get_embedder()
    query_vector = list(embedder.embed([query]))[0].tolist()

    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        chunks.append({
            "text": meta.get("text", ""),
            "source": meta.get("source", ""),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "similarity": round(float(match.score), 4),
        })
    return chunks
