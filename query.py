"""
query.py — Query the existing RAG database (no rebuilding)
===========================================================
Run this after rag_demo.py has already built the ChromaDB collection.
"""

import os
import textwrap
import chromadb
from chromadb.utils import embedding_functions


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    ef = embedding_functions.DefaultEmbeddingFunction()
    existing = [c.name for c in client.list_collections()]

    if "aquabot_docs" not in existing:
        print("ERROR: No collection found. Run rag_demo.py first to build it.")
        exit(1)

    collection = client.get_collection(name="aquabot_docs", embedding_function=ef)
    print(f"Loaded collection: {collection.count()} vectors ready.\n")
    return collection


def retrieve(collection: chromadb.Collection, query: str, top_k: int = 3) -> list[dict]:
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


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    prompt = f"""You are a helpful AquaBot support assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    else:
        return (
            "[No ANTHROPIC_API_KEY — showing raw retrieved chunks]\n\n"
            + "\n\n".join(f"• {c['text'][:200]}..." for c in chunks)
        )


def main():
    print("=" * 60)
    print("  AquaBot RAG — Query Mode")
    print("=" * 60)

    collection = load_collection()

    while True:
        query = input("Ask a question (or 'exit'): ").strip()
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue

        chunks = retrieve(collection, query, top_k=3)

        print("\nSources:")
        for c in chunks:
            print(f"  [{c['similarity']:.2f}] {c['source']} → {c['text'][:80]}...")

        print("\nAnswer:")
        answer = generate_answer(query, chunks)
        print(textwrap.fill(answer, width=70, initial_indent="  ", subsequent_indent="  "))
        print()


if __name__ == "__main__":
    main()
