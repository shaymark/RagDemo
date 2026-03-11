"""
app.py — RAG Web App (FastAPI)
=================================
Admin panel + REST API wrapping the existing RAG pipeline.

Run:
    source venv/bin/activate
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_engine import load_documents, prepare_chunks, build_vector_store, load_collection, retrieve

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SETTINGS_FILE = BASE_DIR / "settings.json"
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AquaBot support assistant. "
    "Answer using ONLY the context below.\n"
    'If the answer is not in the context, say "I don\'t have that information."'
)

# ── Settings helpers ──────────────────────────────────────────────────────────
def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        defaults = {
            "api_key": "",
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "top_k": 3,
        }
        SETTINGS_FILE.write_text(json.dumps(defaults, indent=2))
        return defaults
    return json.loads(SETTINGS_FILE.read_text())


def save_settings(data: dict):
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


# ── Re-indexing helper ────────────────────────────────────────────────────────
def reindex() -> int:
    """Rebuild ChromaDB collection from ./docs/. Returns total chunk count."""
    documents = load_documents(str(DOCS_DIR))
    ids, texts, metadatas = prepare_chunks(documents)
    build_vector_store(ids, texts, metadatas)
    return len(ids)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AquaBot RAG API", version="1.0.0")


@app.on_event("startup")
def startup():
    # Ensure settings.json exists
    load_settings()
    # Build index if collection doesn't exist yet
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if "aquabot_docs" not in existing:
        print("No collection found — building index from ./docs/ ...")
        reindex()
        print("Index ready.")


# Serve static files and the root HTML page
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse)
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Pydantic models ───────────────────────────────────────────────────────────
class SettingsUpdate(BaseModel):
    api_key: str
    system_prompt: str
    top_k: int = 3


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None


# ── Settings endpoints ────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    s = load_settings()
    return {
        "api_key": "****" if s.get("api_key") else "",
        "system_prompt": s.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        "top_k": s.get("top_k", 3),
    }


@app.post("/api/settings")
def post_settings(body: SettingsUpdate):
    current = load_settings()
    # If user sent back the masked value, keep the existing key
    api_key = current["api_key"] if body.api_key == "****" else body.api_key
    save_settings({
        "api_key": api_key,
        "system_prompt": body.system_prompt,
        "top_k": body.top_k,
    })
    return {"success": True}


# ── Documents endpoints ───────────────────────────────────────────────────────
@app.get("/api/documents")
def list_documents():
    docs = []
    for f in sorted(DOCS_DIR.iterdir()):
        if f.suffix == ".txt" and f.is_file():
            docs.append({"name": f.name, "size": f.stat().st_size})
    return {"documents": docs}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    safe_name = Path(file.filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    dest = DOCS_DIR / safe_name
    dest.write_bytes(await file.read())
    chunks_total = reindex()
    return {"success": True, "filename": safe_name, "chunks_total": chunks_total}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
    target = DOCS_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    chunks_total = reindex()
    return {"success": True, "deleted": safe_name, "chunks_total": chunks_total}


# ── Chat endpoint (also the external REST API) ────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    settings = load_settings()
    api_key = settings.get("api_key", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Set it in Settings.",
        )

    top_k = req.top_k if req.top_k is not None else settings.get("top_k", 3)

    # Retrieve relevant chunks from vector store
    collection = load_collection()
    chunks = retrieve(collection, req.query, top_k=top_k)

    # Build prompt with configurable system prompt
    system_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {req.query}\n\n"
        f"ANSWER:"
    )

    # Call Anthropic API
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return {
        "answer": message.content[0].text,
        "sources": chunks,
        "query": req.query,
    }


# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        existing = [c.name for c in client.list_collections()]
        collection_ready = "aquabot_docs" in existing
        vector_count = 0
        if collection_ready:
            ef = embedding_functions.DefaultEmbeddingFunction()
            col = client.get_collection("aquabot_docs", embedding_function=ef)
            vector_count = col.count()
        doc_count = len([f for f in DOCS_DIR.iterdir() if f.suffix == ".txt"])
        return {
            "status": "ok",
            "collection_ready": collection_ready,
            "vector_count": vector_count,
            "doc_count": doc_count,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
