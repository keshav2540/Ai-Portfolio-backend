import logging
import os
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

# Task 1.2: Suppress httpx and httpcore INFO logs to prevent key leaks
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("embedding")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def get_embedding(text: str) -> List[float]:
    """Generates an embedding vector for a single string using Gemini embedding API."""
    if os.getenv("TESTING"):
        return [0.0] * EMBEDDING_DIM
    key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""
    key = key.strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")

    # Task 1.1: Use x-goog-api-key header instead of query parameter
    model_name = EMBEDDING_MODEL if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL.replace("models/", "")
    url = f"{BASE_URL}/models/{model_name}:embedContent"
    headers = {
        "x-goog-api-key": key,
        "Content-Type": "application/json",
    }
    payload = {
        "content": {"parts": [{"text": text}]},
        "outputDimensionality": EMBEDDING_DIM,
    }

    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API failed ({resp.status_code})"
            )
        data = resp.json()
        return data.get("embedding", {}).get("values", [])


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embedding vectors for a list of strings."""
    return [get_embedding(t) for t in texts]
