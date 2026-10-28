import atexit
import os
import re
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import embeddings as embedding

load_dotenv()

COLLECTION_NAME = "resume_knowledge"
LOCAL_QDRANT_DIR = os.path.join(os.path.dirname(__file__), "qdrant_db")

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Returns an active QdrantClient, trying Cloud first and falling back to embedded local storage."""
    global _client
    if _client is not None:
        return _client

    url = os.getenv("Qdrant_url")
    key = os.getenv("Qdrant_api_key")
    is_cloud = bool(os.getenv("VERCEL") or os.getenv("RENDER"))

    if is_cloud:
        if not url or not key:
            raise RuntimeError(
                "Running in cloud environment (Vercel/Render), but Qdrant_url or Qdrant_api_key is missing. "
                "Local embedded Qdrant cannot be used in ephemeral cloud instances. Please configure Qdrant_url and Qdrant_api_key in the dashboard."
            )
        try:
            client = QdrantClient(
                url=url,
                api_key=key,
                timeout=5.0,
                check_compatibility=False,
            )
            # Test ping
            client.get_collections()
            print("[Qdrant] Connected to Qdrant Cloud cluster.")
            _client = client
            return _client
        except Exception as e:
            raise RuntimeError(
                f"Running in cloud environment, but failed to connect to Qdrant Cloud: {e}. "
                "Local embedded Qdrant fallback is disabled on cloud deployments."
            ) from e

    if url and key:
        try:
            client = QdrantClient(
                url=url,
                api_key=key,
                timeout=5.0,
                check_compatibility=False,
            )
            # Test ping
            client.get_collections()
            print("[Qdrant] Connected to Qdrant Cloud cluster.")
            _client = client
            return _client
        except Exception as e:
            print(f"[Qdrant Cloud notice] Cloud endpoint not reachable ({e}). Switching to local embedded vector storage.")

    os.makedirs(LOCAL_QDRANT_DIR, exist_ok=True)
    _client = QdrantClient(path=LOCAL_QDRANT_DIR)
    print(f"[Qdrant] Running local embedded vector database at {LOCAL_QDRANT_DIR}.")
    return _client


def close_client():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(close_client)


_has_indexed_cache: Optional[bool] = None


def init_collection(force_recreate: bool = False):
    """Ensures the resume_knowledge collection exists with proper vector params."""
    global _has_indexed_cache
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        if force_recreate:
            client.delete_collection(COLLECTION_NAME)
            _has_indexed_cache = False
        else:
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=embedding.EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    _has_indexed_cache = False
    print(f"[Qdrant] Created collection '{COLLECTION_NAME}'.")


def has_indexed_data() -> bool:
    """Checks whether the vector database already contains indexed resume data (cached in memory)."""
    if os.getenv("TESTING"):
        return False

    global _has_indexed_cache
    if _has_indexed_cache is not None:
        return _has_indexed_cache

    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            _has_indexed_cache = False
            return False
        info = client.get_collection(COLLECTION_NAME)
        _has_indexed_cache = (info.points_count or 0) > 0
        return _has_indexed_cache
    except Exception:
        return False


def chunk_resume(text: str) -> List[Dict[str, Any]]:
    """
    Intelligently chunks resume text into semantic units (Profile, Skills, individual Projects, Education).
    """
    chunks = []
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    current_section = "OVERVIEW"
    section_map: Dict[str, List[str]] = {}

    KNOWN_SECTIONS = [
        "SUMMARY",
        "FEATURED PROJECTS",
        "PROJECTS",
        "TECHNICAL SKILLS",
        "EDUCATION",
        "MORE PROJECTS",
        "TRAINING",
        "EXPERIENCE",
        "INTERESTS",
    ]

    for line in lines:
        upper = line.upper().strip()
        matched_sec = None
        for s in KNOWN_SECTIONS:
            if upper == s or upper.startswith(s + ":") or upper.startswith(s + " ("):
                matched_sec = "PROJECTS" if "PROJECTS" in s and "MORE" not in s else s
                break

        if matched_sec:
            current_section = matched_sec
            if current_section not in section_map:
                section_map[current_section] = []
            if ":" in line and line.split(":", 1)[0].upper().strip() in KNOWN_SECTIONS:
                after = line.split(":", 1)[1].strip()
                if after:
                    section_map[current_section].append(after)
        else:
            if current_section not in section_map:
                section_map[current_section] = []
            section_map[current_section].append(line)

    order_idx = 0

    # 1. Add overview / contact
    if "OVERVIEW" in section_map:
        overview_text = "\n".join(section_map["OVERVIEW"])
        chunks.append({
            "id": order_idx,
            "section": "Contact & Overview",
            "content": f"Personal Details & Contact:\n{overview_text}",
        })
        order_idx += 1

    # 2. Add summary
    if "SUMMARY" in section_map:
        chunks.append({
            "id": order_idx,
            "section": "Professional Summary",
            "content": "Professional Summary:\n" + "\n".join(section_map["SUMMARY"]),
        })
        order_idx += 1

    # 3. Add education
    if "EDUCATION" in section_map:
        chunks.append({
            "id": order_idx,
            "section": "Education & Academics",
            "content": "Education & Academics:\n" + "\n".join(section_map["EDUCATION"]),
        })
        order_idx += 1

    # 4. Add technical skills (whole block + category chunks)
    if "TECHNICAL SKILLS" in section_map:
        all_skills = "\n".join(section_map["TECHNICAL SKILLS"])
        chunks.append({
            "id": order_idx,
            "section": "Technical Skills Overview",
            "content": f"Technical Skills:\n{all_skills}",
        })
        order_idx += 1

        for skill_line in section_map["TECHNICAL SKILLS"]:
            if ":" in skill_line:
                category, items = skill_line.split(":", 1)
                chunks.append({
                    "id": order_idx,
                    "section": f"Skills - {category.strip()}",
                    "content": f"{category.strip()} Skills: {items.strip()}",
                })
                order_idx += 1

    # 5. Add individual projects
    if "PROJECTS" in section_map:
        current_proj_title = ""
        current_proj_lines = []

        for p_line in section_map["PROJECTS"]:
            if not p_line.startswith("-") and not p_line.startswith("•") and not p_line.startswith("Live:") and not p_line.startswith("Technologies:"):
                if current_proj_title and current_proj_lines:
                    chunks.append({
                        "id": order_idx,
                        "section": f"Project: {current_proj_title}",
                        "content": f"Project: {current_proj_title}\n" + "\n".join(current_proj_lines),
                    })
                    order_idx += 1
                    current_proj_lines = []
                current_proj_title = p_line
            else:
                current_proj_lines.append(p_line)

        if current_proj_title:
            chunks.append({
                "id": order_idx,
                "section": f"Project: {current_proj_title}",
                "content": f"Project: {current_proj_title}\n" + "\n".join(current_proj_lines),
            })
            order_idx += 1

    # 6. Add remaining sections (Interests, Experience, etc.)
    for sec_name, sec_lines in section_map.items():
        if sec_name not in ["OVERVIEW", "SUMMARY", "EDUCATION", "TECHNICAL SKILLS", "PROJECTS"]:
            chunks.append({
                "id": order_idx,
                "section": sec_name.title(),
                "content": f"{sec_name.title()}:\n" + "\n".join(sec_lines),
            })
            order_idx += 1

    return chunks


def ingest_resume_text(raw_text: str) -> int:
    """
    Takes raw resume text, chunks it, generates embeddings, and populates Qdrant.
    Returns the count of points indexed.
    """
    init_collection(force_recreate=True)
    chunks = chunk_resume(raw_text)

    if not chunks:
        raise ValueError("Could not extract any chunks from resume text.")

    client = get_qdrant_client()
    points = []

    print(f"[RAG] Generating embeddings for {len(chunks)} resume chunks...")
    for item in chunks:
        vec = embedding.get_embedding(item["content"])
        points.append(
            PointStruct(
                id=item["id"],
                vector=vec,
                payload={
                    "order_id": item["id"],
                    "section": item["section"],
                    "content": item["content"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    global _has_indexed_cache
    _has_indexed_cache = True
    print(f"[RAG] Successfully stored {len(points)} knowledge points in Qdrant vector database.")
    return len(points)


def query_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    """Queries Qdrant for the most relevant resume chunks for a given query."""
    if not has_indexed_data():
        return []

    client = get_qdrant_client()
    query_vector = embedding.get_embedding(query)

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    return [p.payload["content"] for p in search_result.points if p.payload and "content" in p.payload]


def get_all_indexed_resume() -> str:
    """
    Reconstructs the full resume context from the vector database.
    Allows the application to work completely independently of resume.txt.
    """
    if not has_indexed_data():
        return ""

    client = get_qdrant_client()
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=100,
        with_payload=True,
    )

    sorted_records = sorted(
        records, key=lambda r: (r.payload or {}).get("order_id", 0)
    )

    unique_contents = []
    seen = set()
    for r in sorted_records:
        content = (r.payload or {}).get("content", "")
        if content and content not in seen:
            seen.add(content)
            unique_contents.append(content)

    return "\n\n".join(unique_contents)

