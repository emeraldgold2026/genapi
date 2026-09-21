from typing import Any, Dict, List

from openai import OpenAI

SYSTEM = (
    "You answer questions using ONLY the numbered search results provided. "
    "Synthesize a clear, concise answer and cite results like [1], [2]. "
    "Ignore irrelevant results. If the results do not contain the answer, say so plainly."
)


def build_messages(question: str, matches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    ctx = "\n\n".join(
        f"[{i}] (id={m['id']}, score={m['score']:.3f})\n{m['text'] or m['metadata']}"
        for i, m in enumerate(matches, 1)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nSearch results:\n{ctx}"},
    ]


class Refiner:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def refine(self, question: str, matches: List[Dict[str, Any]]) -> str:
        if not matches:
            return "No matching results were found in the index."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_messages(question, matches),
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
