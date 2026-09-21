from typing import Any, Dict, Optional


class SearchService:
    def __init__(self, store, embedder, refiner, top_k: int = 5):
        self.store, self.embedder, self.refiner, self.top_k = store, embedder, refiner, top_k

    def search(self, query: str, top_k: Optional[int] = None, flt: Optional[dict] = None):
        return self.store.query(self.embedder.embed(query), top_k or self.top_k, flt)

    def chat(self, message: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        matches = self.search(message, top_k)
        warning = None
        try:
            answer = self.refiner.refine(message, matches)
        except Exception as e:  # fall back to raw matches
            answer, warning = "", f"Refinement failed: {e}"
        return {"answer": answer, "matches": matches, "warning": warning}
