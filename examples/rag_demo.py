"""
RAG (Retrieval-Augmented Generation) Demo
==========================================
This script demonstrates the complete RAG pipeline end-to-end.
It is an EXAMPLE script — core logic lives in rag_engine.py.

  1. INGEST  — load text documents and split into chunks
  2. EMBED   — convert chunks to semantic vector embeddings
  3. STORE   — save vectors in ChromaDB (a local vector database)
  4. RETRIEVE — given a query, find the most relevant chunks
  5. GENERATE — pass retrieved chunks + query to an LLM

Run from the project root:
    source venv/bin/activate
    python3 examples/rag_demo.py
"""

import os
import sys
import textwrap

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_engine import (
    load_documents,
    chunk_text,
    prepare_chunks,
    build_vector_store,
    retrieve,
)


# ─────────────────────────────────────────────
# STEP 5: GENERATE (demo-only, uses env var for API key)
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
        return (
            "[No ANTHROPIC_API_KEY found — showing retrieved context instead]\n\n"
            + "\n\n".join(f"• {c['text'][:200]}..." for c in chunks)
        )


# ─────────────────────────────────────────────
# MAIN DEMO
# ─────────────────────────────────────────────

def demo():
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")

    print("=" * 60)
    print("  RAG DEMO — AquaBot Knowledge Base")
    print("=" * 60)

    print("\n[1] Loading documents from ./docs/")
    docs = load_documents(docs_dir)

    print("\n[2] Chunking documents...")
    ids, texts, metadatas = prepare_chunks(docs)

    print("\n[3] Building vector store (ChromaDB)...")
    collection = build_vector_store(ids, texts, metadatas)

    queries = [
        "How much does the Smart Feeder cost?",
        "My fish tank has high ammonia, what should I do?",
        "Who founded AquaBot and when?",
        "What is the difference between free and Pro subscription?",
        "Does AquaBot support reef tanks?",
        "how much Pro Monthly subscription costs?",
    ]

    print("\n" + "=" * 60)
    print("  RETRIEVAL + GENERATION DEMO")
    print("=" * 60)

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

        chunks = retrieve(collection, query, top_k=3)

        print(f"\nTop {len(chunks)} retrieved chunks:")
        for c in chunks:
            print(f"  [{c['similarity']:.2f}] {c['source']} → {c['text'][:80]}...")

        print(f"\nANSWER:")
        answer = generate_answer(query, chunks)
        print(textwrap.fill(answer, width=70, initial_indent="  ", subsequent_indent="  "))

    print(f"\n{'='*60}")
    print("  ChromaDB persisted to ./chroma_db/")
    print("=" * 60)


if __name__ == "__main__":
    demo()
