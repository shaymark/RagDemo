# AquaBot RAG

A full-stack **Retrieval-Augmented Generation (RAG)** system with a web admin panel and REST API, built with FastAPI and ChromaDB.

## What it does

Lets you upload a knowledge base of `.txt` documents, embed them into a vector database, and query them through an LLM (Claude) that answers only from your documents — not from general training data.

---

## Project Structure

```
rag/
├── app.py               # FastAPI server — admin panel + REST API
├── rag_engine.py        # Core RAG functions (embed, store, retrieve)
├── requirements.txt     # Python dependencies
├── settings.json        # Auto-created — stores API key & config (gitignored)
├── docs/                # Your knowledge base (.txt files)
├── chroma_db/           # Vector database — auto-created (gitignored)
├── static/
│   └── index.html       # Admin panel UI
├── venv/                # Python virtual environment (gitignored)
└── examples/
    ├── rag_demo.py      # CLI demo — builds DB and runs sample queries
    └── query.py         # CLI interactive query mode
```

---

## Setup

### 1. Clone and enter the project
```bash
cd /path/to/rag
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the admin panel
```
http://localhost:8000
```

---

## Admin Panel

Three tabs:

| Tab | What you can do |
|-----|-----------------|
| **⚙️ Settings** | Set your Anthropic API key, edit the system prompt, change Top K |
| **📄 Documents** | Upload `.txt` files to the knowledge base, delete existing ones. Re-indexing happens automatically. |
| **💬 Chat** | Test the RAG chatbot directly in the browser |

---

## REST API

The `/api/chat` endpoint can be called from any external app:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How much does the Smart Feeder cost?"}'
```

**Response:**
```json
{
  "answer": "The Smart Feeder costs $49 as a one-time purchase.",
  "sources": [
    { "source": "pricing.txt", "similarity": 0.91, "text": "..." }
  ],
  "query": "How much does the Smart Feeder cost?"
}
```

### All endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Admin panel UI |
| `GET` | `/api/health` | Server and collection status |
| `GET` | `/api/settings` | Get current settings (API key masked) |
| `POST` | `/api/settings` | Save settings |
| `GET` | `/api/documents` | List documents with file sizes |
| `POST` | `/api/documents/upload` | Upload a `.txt` file and re-index |
| `DELETE` | `/api/documents/{filename}` | Delete a document and re-index |
| `POST` | `/api/chat` | Query the RAG — returns answer + sources |

---

## How RAG works

```
Your documents (.txt)
       ↓
  Split into chunks (400 chars, 80-char overlap)
       ↓
  Embed with all-MiniLM-L6-v2 (384-dim vectors)
       ↓
  Store in ChromaDB (local vector database)
       ↓
  Query → embed → cosine similarity search → top-K chunks
       ↓
  Chunks + question → Claude → answer grounded in your docs
```

---

## Example scripts

Run the full pipeline demo from scratch:
```bash
source venv/bin/activate
python3 examples/rag_demo.py
```

Run the interactive query CLI:
```bash
source venv/bin/activate
python3 examples/query.py
```

---

## Configuration

Settings are saved to `settings.json` (auto-created, gitignored):

| Field | Default | Description |
|-------|---------|-------------|
| `api_key` | `""` | Your Anthropic API key (`sk-ant-...`) |
| `system_prompt` | AquaBot prompt | Instructions sent to the LLM on every request |
| `top_k` | `3` | Number of document chunks retrieved per query |

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
