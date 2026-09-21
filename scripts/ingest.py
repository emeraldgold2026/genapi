"""Load a JSONL file ({"id","text", ...metadata}) into the index.

Usage: python -m scripts.ingest data.jsonl
"""
import json
import sys


from app.config import load_settings
from app.embeddings import Embedder
from app.pinecone_client import PineconeStore


def main(path: str, batch: int = 50) -> None:
    s = load_settings()
    store = PineconeStore(s.pinecone_api_key, s.pinecone_host, s.pinecone_namespace, s.text_field)
    emb = Embedder(store.pc, s.embed_model)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        vecs = emb.embed_many([r["text"] for r in chunk])
        store.upsert([
            {"id": str(r["id"]), "values": v,
             "metadata": {**{k: x for k, x in r.items() if k != "id"}, s.text_field: r["text"]}}
            for r, v in zip(chunk, vecs)
        ])
    print(f"Upserted {len(rows)} records")


if __name__ == "__main__":
    main(sys.argv[1])
