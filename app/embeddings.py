from typing import List

from pinecone import Pinecone


class Embedder:
    """Embeds text with Pinecone's hosted model so vectors match the index.

    The index was built with llama-text-embed-v2 (1024-d): queries use
    input_type=query, stored documents use input_type=passage.
    """

    def __init__(self, pc: Pinecone, model: str):
        self.pc = pc
        self.model = model

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        res = self.pc.inference.embed(
            model=self.model,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        return [list(r["values"]) for r in res]

    def embed(self, text: str) -> List[float]:
        return self._embed([text], "query")[0]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), 90):  # inference API batch limit
            out += self._embed(texts[i:i + 90], "passage")
        return out
