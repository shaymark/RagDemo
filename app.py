"""
app.py — RAG Web App (FastAPI)
=================================
Admin panel + REST API wrapping the existing RAG pipeline.
Vector store: Pinecone (cloud). File storage: Supabase (cloud, optional).

Run:
    source venv/bin/activate
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import hashlib
import json
import os
import secrets
from pathlib import Path

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from supabase import create_client

from rag_engine import (
    load_documents,
    prepare_chunks,
    build_vector_store,
    get_index,
    retrieve,
    upsert_document,
    delete_document_vectors,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SETTINGS_FILE = BASE_DIR / "settings.json"
DOCS_DIR = BASE_DIR / "docs"
STATIC_DIR = BASE_DIR / "static"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AquaBot support assistant. "
    "Answer using ONLY the context below.\n"
    "if the user will ask question in different language please answer to him in his language.\n"
    'If the answer is not in the context, say "I don\'t have that information."'
)


# ── Settings helpers ──────────────────────────────────────────────────────────
def load_settings() -> dict:
    # Seed api_key from environment variable if available
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not SETTINGS_FILE.exists():
        defaults = {
            "api_key": env_key,
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
            "top_k": 3,
            "session_secret": secrets.token_hex(32),
        }
        SETTINGS_FILE.write_text(json.dumps(defaults, indent=2))
        return defaults
    data = json.loads(SETTINGS_FILE.read_text())
    # If settings file has no key but env var is set, use the env var
    if not data.get("api_key") and env_key:
        data["api_key"] = env_key
    # Backfill session_secret for existing settings files
    if not data.get("session_secret"):
        data["session_secret"] = secrets.token_hex(32)
        SETTINGS_FILE.write_text(json.dumps(data, indent=2))
    return data


def save_settings(data: dict):
    # Preserve fields not managed by the settings form (e.g. session_secret)
    current = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
    current.update(data)
    SETTINGS_FILE.write_text(json.dumps(current, indent=2))


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _hash_password(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


# Resolve session secret: env var > settings.json > freshly generated
_bootstrap = load_settings()
SESSION_SECRET = os.environ.get(
    "SESSION_SECRET_KEY",
    _bootstrap.get("session_secret") or secrets.token_hex(32),
)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
_ADMIN_PW_HASH = _hash_password(os.environ.get("ADMIN_PASSWORD", "admin"))


# ── Supabase storage helper ───────────────────────────────────────────────────
def _get_storage():
    """
    Returns a Supabase storage bucket client, or None if env vars are not set.
    When None, file operations fall back to ./docs/ on local disk.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    bucket = os.environ.get("SUPABASE_BUCKET", "docs")
    return create_client(url, key).storage.from_(bucket)


def _sf_name(f) -> str:
    """Get filename from a Supabase file object (handles both dict and object)."""
    return f.name if hasattr(f, "name") else f["name"]


def _sf_size(f) -> int:
    """Get file size in bytes from a Supabase file object."""
    meta = f.metadata if hasattr(f, "metadata") else f.get("metadata", {})
    return (meta or {}).get("size", 0)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="AquaBot RAG API", version="1.0.0")

IS_PROD = os.environ.get("RENDER", "").lower() == "true"
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    https_only=IS_PROD,
    same_site="lax",
)


# ── Auth dependency ───────────────────────────────────────────────────────────
def require_auth(request: Request):
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=401, detail="Not authenticated")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    load_settings()

    pinecone_key = os.environ.get("PINECONE_API_KEY", "")
    if not pinecone_key:
        print("WARNING: PINECONE_API_KEY not set — skipping startup index check.")
        return

    try:
        storage = _get_storage()

        # Seed Supabase bucket from git sample files on first deploy
        if storage:
            remote_files = {_sf_name(f) for f in storage.list()}
            if not remote_files:
                print("Supabase bucket is empty — seeding from ./docs/ ...")
                for f in sorted(DOCS_DIR.iterdir()):
                    if f.suffix == ".txt":
                        storage.upload(f.name, f.read_bytes(), {"upsert": "true"})
                        print(f"  Seeded: {f.name}")

        # Build Pinecone index if empty or missing
        try:
            index = get_index()
            stats = index.describe_index_stats()
            vector_count = stats.total_vector_count
        except RuntimeError:
            vector_count = 0

        if vector_count == 0:
            print("Pinecone index is empty — building ...")
            _build_index_from_source(storage)
            print("Index ready.")
        else:
            print(f"Pinecone index ready ({vector_count} vectors).")

    except Exception as e:
        print(f"WARNING: Startup index check failed: {e}")


def _build_index_from_source(storage) -> None:
    """Build (or rebuild) the full Pinecone index from Supabase or local disk."""
    if storage:
        files = [f for f in storage.list() if _sf_name(f).endswith(".txt")]
        documents = []
        for f in files:
            name = _sf_name(f)
            text = storage.download(name).decode("utf-8")
            documents.append({
                "id": Path(name).stem,
                "text": text,
                "source": name,
            })
    else:
        documents = load_documents(str(DOCS_DIR))

    if documents:
        ids, texts, metadatas = prepare_chunks(documents)
        build_vector_store(ids, texts, metadatas)


# ── Static files + page routes ────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "chat.html"))


@app.get("/login")
def login_page():
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.get("/admin")
def admin_page(request: Request):
    if not request.session.get("authenticated"):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Pydantic models ───────────────────────────────────────────────────────────
class SettingsUpdate(BaseModel):
    api_key: str
    system_prompt: str
    top_k: int = 3


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None


class DocumentUpdate(BaseModel):
    content: str


class LoginRequest(BaseModel):
    username: str
    password: str


# ── Auth endpoints ────────────────────────────────────────────────────────────
@app.post("/api/login")
def api_login(body: LoginRequest, request: Request):
    username_ok = secrets.compare_digest(body.username, ADMIN_USERNAME)
    password_ok = secrets.compare_digest(
        _hash_password(body.password), _ADMIN_PW_HASH
    )
    if username_ok and password_ok:
        request.session["authenticated"] = True
        request.session["username"] = body.username
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/api/logout")
def api_logout(request: Request):
    request.session.clear()
    return {"success": True}


@app.get("/api/auth/status")
def auth_status(request: Request):
    return {"authenticated": bool(request.session.get("authenticated"))}


# ── Settings endpoints ────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings(_: None = Depends(require_auth)):
    s = load_settings()
    return {
        "api_key": "****" if s.get("api_key") else "",
        "system_prompt": s.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        "top_k": s.get("top_k", 3),
    }


@app.post("/api/settings")
def post_settings(body: SettingsUpdate, _: None = Depends(require_auth)):
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
def list_documents(_: None = Depends(require_auth)):
    storage = _get_storage()
    if storage:
        docs = [
            {"name": _sf_name(f), "size": _sf_size(f)}
            for f in storage.list()
            if _sf_name(f).endswith(".txt")
        ]
        docs.sort(key=lambda d: d["name"])
    else:
        docs = [
            {"name": f.name, "size": f.stat().st_size}
            for f in sorted(DOCS_DIR.iterdir())
            if f.suffix == ".txt" and f.is_file()
        ]
    return {"documents": docs}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...), _: None = Depends(require_auth)
):
    safe_name = Path(file.filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    content_bytes = await file.read()
    content_text = content_bytes.decode("utf-8")

    storage = _get_storage()
    if storage:
        storage.upload(safe_name, content_bytes, {"upsert": "true"})
    else:
        (DOCS_DIR / safe_name).write_bytes(content_bytes)

    chunks = upsert_document(safe_name, content_text)
    return {"success": True, "filename": safe_name, "chunks_total": chunks}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str, _: None = Depends(require_auth)):
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    storage = _get_storage()
    if storage:
        storage.remove([safe_name])
    else:
        target = DOCS_DIR / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        target.unlink()

    delete_document_vectors(safe_name)
    return {"success": True, "deleted": safe_name}


@app.get("/api/documents/{filename}/content")
def get_document_content(filename: str, _: None = Depends(require_auth)):
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    storage = _get_storage()
    if storage:
        try:
            content = storage.download(safe_name).decode("utf-8")
        except Exception:
            raise HTTPException(status_code=404, detail="File not found")
    else:
        target = DOCS_DIR / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        content = target.read_text()

    return {"filename": safe_name, "content": content}


@app.put("/api/documents/{filename}")
def update_document(
    filename: str, body: DocumentUpdate, _: None = Depends(require_auth)
):
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    storage = _get_storage()
    if storage:
        storage.upload(safe_name, body.content.encode("utf-8"), {"upsert": "true"})
    else:
        target = DOCS_DIR / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        target.write_text(body.content)

    chunks = upsert_document(safe_name, body.content)
    return {"success": True, "filename": safe_name, "chunks_total": chunks}


@app.get("/api/documents/{filename}")
def download_document(filename: str, _: None = Depends(require_auth)):
    safe_name = Path(filename).name
    if not safe_name.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    storage = _get_storage()
    if storage:
        try:
            data = storage.download(safe_name)
        except Exception:
            raise HTTPException(status_code=404, detail="File not found")
        return Response(
            content=data,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={safe_name}"},
        )
    else:
        target = DOCS_DIR / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(
            str(target),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={safe_name}"},
        )


# ── Chat endpoint (public — no auth required) ─────────────────────────────────
@app.post("/api/chat")
def chat(req: ChatRequest):
    settings = load_settings()
    api_key = settings.get("api_key", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key not configured. Set it in Settings.",
        )

    pinecone_key = os.environ.get("PINECONE_API_KEY", "")
    if not pinecone_key:
        raise HTTPException(
            status_code=400,
            detail="PINECONE_API_KEY not configured.",
        )

    top_k = req.top_k if req.top_k is not None else settings.get("top_k", 3)

    # Retrieve relevant chunks from Pinecone
    index = get_index()
    chunks = retrieve(index, req.query, top_k=top_k)

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


# ── Health endpoint (public) ───────────────────────────────────────────────────
@app.get("/api/health")
def health():
    try:
        pinecone_key = os.environ.get("PINECONE_API_KEY", "")
        if not pinecone_key:
            return {"status": "error", "detail": "PINECONE_API_KEY not set"}

        index = get_index()
        stats = index.describe_index_stats()
        vector_count = stats.total_vector_count

        storage = _get_storage()
        if storage:
            doc_count = sum(1 for f in storage.list() if _sf_name(f).endswith(".txt"))
        else:
            doc_count = len([f for f in DOCS_DIR.iterdir() if f.suffix == ".txt"])

        return {
            "status": "ok",
            "collection_ready": vector_count > 0,
            "vector_count": vector_count,
            "doc_count": doc_count,
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
