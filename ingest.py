#!/usr/bin/env python3
"""
One-Time Ingestion Script:
Reads resume.txt, generates vector embeddings using Gemini, and stores the knowledge in Qdrant.
After running this script once, your resume data is permanently stored in the vector DB,
and resume.txt can be safely deleted.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
import rag

DEFAULT_RESUME = os.path.join(os.path.dirname(__file__), "resume.txt")


def run_ingestion(resume_path: str = DEFAULT_RESUME):
    if not os.path.exists(resume_path):
        print(f"[Error] Resume file not found at: {resume_path}")
        print("Please provide a valid resume file or place resume.txt in the backend folder.")
        sys.exit(1)

    print(f"==================================================")
    print(f"  Ingesting Resume into Qdrant Vector Database")
    print(f"==================================================")
    print(f"Reading from: {resume_path}")

    with open(resume_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Total characters: {len(text)}")
    print("Chunking and generating embeddings via Gemini...")

    count = rag.ingest_resume_text(text)

    print(f"\n[Success] Ingested {count} semantic chunks into Qdrant vector database.")

    # Verification test query
    test_query = "What projects has Keshav built?"
    print(f"\nVerifying retrieval with test query: '{test_query}'...")
    chunks = rag.query_relevant_chunks(test_query, top_k=2)

    for i, c in enumerate(chunks, 1):
        preview = c.replace("\n", " ")[:90]
        print(f"  Result {i}: {preview}...")

    print(f"\n==================================================")
    print(f"  Indexing Complete!")
    print(f"  The vector database is ready.")
    print(f"  You can now safely delete '{os.path.basename(resume_path)}'.")
    print(f"==================================================")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
    run_ingestion(path)

