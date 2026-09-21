from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

from .config import load_settings
from .embeddings import Embedder
from .pinecone_client import DimensionMismatch, PineconeStore
from .refiner import Refiner
from .service import SearchService

STATIC = Path(__file__).resolve().parent.parent / "static"


class QueryIn(BaseModel):
    query: str
    top_k: Optional[int] = None
    filter: Optional[Dict[str, Any]] = None


class ChatIn(BaseModel):
    message: str
    top_k: Optional[int] = None


class UpsertItem(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class UpsertIn(BaseModel):
    items: List[UpsertItem]


def create_app(service: Optional[SearchService] = None, store=None, embedder=None) -> FastAPI:
    app = FastAPI(title="Pinecone Semantic Chat")

    if service is None:
        s = load_settings()
        oa = OpenAI(api_key=s.openai_api_key)
        store = PineconeStore(s.pinecone_api_key, s.pinecone_host, s.pinecone_namespace, s.text_field)
        embedder = Embedder(store.pc, s.embed_model)
        service = SearchService(store, embedder, Refiner(oa, s.chat_model), s.top_k)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        try:
            stats = service.store.stats()
            dim = service.store.check_dimension(len(service.embedder.embed("dimension check")))
            return {"ok": True, "dimension": dim, "total_vectors": stats.get("total_vector_count")}
        except DimensionMismatch as e:
            raise HTTPException(status_code=409, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/query")
    def query(body: QueryIn):
        try:
            return {"matches": service.search(body.query, body.top_k, body.filter)}
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/chat")
    def chat(body: ChatIn):
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message is empty")
        try:
            return service.chat(body.message, body.top_k)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    @app.post("/api/upsert")
    def upsert(body: UpsertIn):
        texts = [i.text for i in body.items]
        vecs = service.embedder.embed_many(texts)
        field = service.store.text_field
        n = service.store.upsert([
            {"id": i.id, "values": v, "metadata": {**i.metadata, field: i.text}}
            for i, v in zip(body.items, vecs)
        ])
        return {"upserted": n}

    return app


_app = None


def __getattr__(name):  # `uvicorn app.main:app` builds the app on first access
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
