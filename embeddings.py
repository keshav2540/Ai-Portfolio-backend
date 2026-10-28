"""
Embeddings Module (Sole remaining Google API dependency)

IMPORTANT NOTE:
Groq does NOT provide an embeddings API.
This module uses Google's Gemini Embedding REST API (models/gemini-embedding-001)
to generate 768-dimensional vector embeddings for Qdrant Cloud similarity search.

Why this Google dependency is kept:
Your Qdrant Cloud collection ("resume_knowledge") is already indexed with 768-dimensional
Gemini vectors. Changing this embedding model would invalidate the existing index and
require complete re-indexing.

Local Alternative Suggestion:
- fastembed (BAAI/bge-small-en-v1.5) or sentence-transformers (all-MiniLM-L6-v2)
- Tradeoff: Completely free & runs locally without API keys, but outputs 384 dimensions,
  so Qdrant collection must be re-created and resume re-indexed via ingest.py.
"""
import logging
import os
from typing import List
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("embeddings")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def get_embedding(text: str) -> List[float]:
    """Generates a 768-dim embedding vector using Gemini REST API."""
    if os.getenv("TESTING"):
        return [0.0] * EMBEDDING_DIM
    key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""
    key = key.strip()
    if not key:
        raise ValueError("GEMINI_API_KEY (or gemini_api_key) is required for Qdrant vector embeddings.")

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

