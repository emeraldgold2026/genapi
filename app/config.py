import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    pinecone_host: str
    pinecone_namespace: str
    text_field: str
    openai_api_key: str
    embed_model: str
    chat_model: str
    top_k: int


def load_settings() -> Settings:
    def need(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise RuntimeError(f"Missing required setting {name} (see .env.example)")
        return value

    return Settings(
        pinecone_api_key=need("PINECONE_API_KEY"),
        pinecone_host=need("PINECONE_HOST"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
        text_field=os.getenv("PINECONE_TEXT_FIELD", "text").strip() or "text",
        openai_api_key=need("OPENAI_API_KEY"),
        embed_model=os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2"),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        top_k=int(os.getenv("TOP_K", "5")),
    )
