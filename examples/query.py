"""
query.py — Interactive query mode (example script)
====================================================
Query the existing RAG database without rebuilding it.
Core logic lives in rag_engine.py.

Run from the project root:
    source venv/bin/activate
    python3 examples/query.py
"""

import os
import sys
import textwrap

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_engine import load_collection, retrieve


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

    try:
        collection = load_collection()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Loaded collection: {collection.count()} vectors ready.\n")

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
