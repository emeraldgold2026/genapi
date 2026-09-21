# Pinecone Semantic Chat

FastAPI app + chat UI for semantic search over a Pinecone index, with OpenAI refining the results.

## Run
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    cp .env.example .env   # fill in keys
    .venv/bin/uvicorn app.main:app --port 8000   # open http://localhost:8000

## API
- `GET /api/health` – connectivity + index dimension check
- `POST /api/query` `{query, top_k?, filter?}` – raw semantic search
- `POST /api/chat` `{message, top_k?}` – search, then OpenAI refines -> `{answer, matches, warning}`
- `POST /api/upsert` `{items:[{id,text,metadata}]}` – embed + store
- `python -m scripts.ingest file.jsonl` – bulk load

Embeddings use Pinecone's hosted `llama-text-embed-v2` (what the index was built with); refinement uses `OPENAI_CHAT_MODEL`.
Tests: `.venv/bin/python -m pytest`

The application queries apple 10 K document and returns response using symantic search.