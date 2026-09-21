from typing import Any, Dict, List, Optional

from pinecone import Pinecone


class DimensionMismatch(RuntimeError):
    pass


class PineconeStore:
    def __init__(self, api_key: str, host: str, namespace: str = "", text_field: str = "text"):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(host=host)
        self.namespace = namespace
        self.text_field = text_field

    def stats(self) -> Dict[str, Any]:
        s = self.index.describe_index_stats()
        return s.to_dict() if hasattr(s, "to_dict") else dict(s)

    def check_dimension(self, embed_dim: int) -> int:
        dim = self.stats().get("dimension")
        if dim is not None and dim != embed_dim:
            raise DimensionMismatch(
                f"Index dimension is {dim} but the embedding model produces {embed_dim}. "
                "Set OPENAI_EMBED_MODEL to the model used to build the index."
            )
        return dim

    def query(self, vector: List[float], top_k: int = 5, flt: Optional[dict] = None) -> List[Dict[str, Any]]:
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=flt,
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out = []
        for m in matches:
            get = m.get if isinstance(m, dict) else lambda k, d=None: getattr(m, k, d)
            md = dict(get("metadata") or {})
            out.append({
                "id": get("id"),
                "score": float(get("score") or 0.0),
                "text": str(md.get(self.text_field, "")),
                "metadata": md,
            })
        return out

    def upsert(self, items: List[Dict[str, Any]]) -> int:
        """items: {id, values, metadata}"""
        self.index.upsert(vectors=items, namespace=self.namespace)
        return len(items)
