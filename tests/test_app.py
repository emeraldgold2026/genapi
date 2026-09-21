from fastapi.testclient import TestClient

from app.main import create_app
from app.pinecone_client import DimensionMismatch
from app.refiner import Refiner, build_messages
from app.service import SearchService

MATCHES = [{"id": "a", "score": 0.9, "text": "Paris is the capital of France.", "metadata": {}}]


class FakeStore:
    text_field = "text"
    dim = 3

    def stats(self):
        return {"dimension": self.dim, "total_vector_count": 1}

    def check_dimension(self, n):
        if n != self.dim:
            raise DimensionMismatch("bad dim")
        return self.dim

    def query(self, v, k, f=None):
        return MATCHES

    def upsert(self, items):
        return len(items)


class FakeEmbedder:
    def __init__(self, n=3):
        self.n = n

    def embed(self, t):
        return [0.1] * self.n

    def embed_many(self, ts):
        return [[0.1] * self.n for _ in ts]


class FakeRefiner:
    def refine(self, q, m):
        return "Paris [1]"


class BoomRefiner:
    def refine(self, q, m):
        raise RuntimeError("openai down")


def client(refiner=None, dim=3):
    store = FakeStore()
    store.dim = dim
    svc = SearchService(store, FakeEmbedder(), refiner or FakeRefiner())
    return TestClient(create_app(svc))


def test_chat_returns_refined_answer_and_matches():
    j = client().post("/api/chat", json={"message": "capital of France?"}).json()
    assert j["answer"] == "Paris [1]" and j["matches"] == MATCHES and j["warning"] is None


def test_chat_falls_back_to_raw_matches_when_refiner_fails():
    j = client(BoomRefiner()).post("/api/chat", json={"message": "x"}).json()
    assert j["answer"] == "" and j["matches"] == MATCHES and "openai down" in j["warning"]


def test_raw_query_and_upsert():
    c = client()
    assert c.post("/api/query", json={"query": "x"}).json()["matches"] == MATCHES
    assert c.post("/api/upsert", json={"items": [{"id": "1", "text": "hi"}]}).json() == {"upserted": 1}


def test_empty_message_rejected():
    assert client().post("/api/chat", json={"message": "  "}).status_code == 400


def test_health_reports_dimension_mismatch():
    assert client(dim=1536).get("/api/health").status_code == 409
    assert client().get("/api/health").json()["ok"] is True


def test_refiner_prompt_contains_question_and_results():
    msgs = build_messages("Q?", MATCHES)
    assert "Q?" in msgs[1]["content"] and "Paris" in msgs[1]["content"]


def test_refiner_short_circuits_without_matches():
    assert "No matching" in Refiner(None, "m").refine("q", [])
