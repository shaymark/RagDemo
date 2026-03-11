"""
RAG (Retrieval-Augmented Generation) Demo
==========================================
This script demonstrates a complete RAG pipeline:
  1. INGEST  — load text documents and split into chunks
  2. EMBED   — convert chunks to semantic vector embeddings
  3. STORE   — save vectors in ChromaDB (a local vector database)
  4. RETRIEVE — given a query, find the most relevant chunks
  5. GENERATE — (optional) pass retrieved chunks + query to an LLM
"""

import os
import textwrap
import chromadb
from chromadb.utils import embedding_functions


# ─────────────────────────────────────────────
# STEP 1: LOAD AND CHUNK DOCUMENTS
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

    Why overlap? So that sentences near a boundary appear in two chunks,
    preventing a relevant sentence from being cut in half and missed.

    chunk_size: max characters per chunk
    overlap:    how many characters the next chunk shares with the previous
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # slide window forward
    return chunks


def prepare_chunks(documents: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """
    Turn documents into (ids, texts, metadatas) ready for ChromaDB.
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
# STEP 2 & 3: EMBED AND STORE IN CHROMADB
# ─────────────────────────────────────────────
def build_vector_store(ids: list[str], texts: list[str], metadatas: list[dict]) -> chromadb.Collection:
    """
    Create a ChromaDB collection and add all chunks.

    ChromaDB uses the `all-MiniLM-L6-v2` sentence-transformer model by default
    to convert text → 384-dimensional vectors. These vectors capture *meaning*,
    so "How much does the feeder cost?" and "Smart Feeder price is $49" will
    have similar vectors even though they share no exact words.
    """
    # Persistent client saves the DB to disk so it survives restarts
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection to start fresh on each run
    try:
        client.delete_collection("aquabot_docs")
    except Exception:
        pass

    # Create a collection — think of it like a table in a SQL database
    # The embedding function runs automatically on every add() and query()
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name="aquabot_docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )

    print(f"\n  Embedding and storing {len(ids)} chunks...")
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"  Done! Collection '{collection.name}' has {collection.count()} vectors.")
    return collection


# ─────────────────────────────────────────────
# STEP 4: RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────
def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[dict]:
    """
    Convert the query to a vector and find the top_k most similar chunks.

    ChromaDB computes cosine similarity between the query vector and every
    stored vector, then returns the closest matches.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "similarity": 1 - results["distances"][0][i],  # distance → similarity
        })
    return retrieved


# ─────────────────────────────────────────────
# STEP 5: GENERATE (uses Claude API if key available)
# ─────────────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""You are a helpful AquaBot support assistant

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def generate_answer(query: str, chunks: list[dict]) -> str:
    prompt = build_prompt(query, chunks)
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    else:
        # No LLM available — return the raw context so you can see what would be sent
        return (
            "[No ANTHROPIC_API_KEY found — showing retrieved context instead]\n\n"
            + "\n\n".join(f"• {c['text'][:200]}..." for c in chunks)
        )


# ─────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────
def demo():
    print("=" * 60)
    print("  RAG DEMO — AquaBot Knowledge Base")
    print("=" * 60)

    # ── INGEST ──
    print("\n[1] Loading documents from ./docs/")
    docs = load_documents("./docs")

    print("\n[2] Chunking documents...")
    ids, texts, metadatas = prepare_chunks(docs)

    # ── EMBED + STORE ──
    print("\n[3] Building vector store (ChromaDB)...")
    collection = build_vector_store(ids, texts, metadatas)

    # ── DEMO QUERIES ──
    queries = [
        "How much does the Smart Feeder cost?",
        "My fish tank has high ammonia, what should I do?",
        "Who founded AquaBot and when?",
        "What is the difference between free and Pro subscription?",
        "Does AquaBot support reef tanks?",
        "how much Pro Monthly subscription costs?"
    ]

    print("\n" + "=" * 60)
    print("  RETRIEVAL + GENERATION DEMO")
    print("=" * 60)

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

        # Retrieve
        chunks = retrieve(collection, query, top_k=3)

        print(f"\nTop {len(chunks)} retrieved chunks:")
        for c in chunks:
            print(f"  [{c['similarity']:.2f}] {c['source']} → {c['text'][:80]}...")

        # Generate
        print(f"\nANSWER:")
        answer = generate_answer(query, chunks)
        print(textwrap.fill(answer, width=70, initial_indent="  ", subsequent_indent="  "))

    print(f"\n{'='*60}")
    print("  ChromaDB persisted to ./chroma_db/")
    print("  You can query it again without re-embedding — fast!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
