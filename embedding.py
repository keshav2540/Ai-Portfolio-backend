import os
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def get_embedding(text: str) -> List[float]:
    """Generates an embedding vector for a single string using Gemini embedding API."""
    if not GEMINI_API_KEY:
        raise ValueError("gemini_api_key is not set in environment.")

    url = f"{BASE_URL}/{EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    payload = {
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": EMBEDDING_DIM,
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API failed ({resp.status_code}): {resp.text}"
            )
        data = resp.json()
        return data.get("embedding", {}).get("values", [])


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embedding vectors for a list of strings."""
    return [get_embedding(t) for t in texts]

