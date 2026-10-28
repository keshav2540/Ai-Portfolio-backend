"""MongoDB & resilient in-memory session memory: chat messages and JD match results, keyed by session id (sid)."""

import collections
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

load_dotenv()

# In-memory fast cache and resilient fallback
_local_messages = collections.defaultdict(list)
_local_matches = collections.defaultdict(list)

_client = None
_db = None
messages = None
matches = None

mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("DB_NAME") or "ai_portfolio"
is_cloud = bool(os.getenv("VERCEL") or os.getenv("RENDER"))

if os.getenv("TESTING"):
    # Fast in-memory mode for testing without external network dependencies
    pass
elif is_cloud and not mongo_uri:
    print(
        "[Store WARNING] Running on cloud (Vercel/Render) but MONGO_URI is not set. "
        "Persistent database storage will not work; falling back to ephemeral in-memory store."
    )
else:
    uri = mongo_uri or "mongodb://localhost:27017"
    try:
        _client = MongoClient(
            uri,
            connectTimeoutMS=2000,
            serverSelectionTimeoutMS=2000,
        )
        _db = _client[db_name]
        messages = _db["messages"]
        matches = _db["matches"]
    except Exception as e:
        print(f"[Store notice] MongoDB connection init delayed or using local memory: {e}")


def init():
    global messages, matches
    try:
        if messages is not None and matches is not None:
            messages.create_index([("sid", ASCENDING), ("ts", ASCENDING)])
            matches.create_index([("sid", ASCENDING), ("ts", DESCENDING)])
    except Exception as e:
        print(f"[Store notice] Could not create Mongo indexes ({e}), proceeding with memory fallback.")
        messages = None
        matches = None


def add_exchange(sid, user_text, bot_text):
    now = time.time()
    # Always keep in local memory for instant zero-latency conversation context
    _local_messages[sid].append({"role": "user", "content": user_text, "ts": now})
    _local_messages[sid].append({"role": "assistant", "content": bot_text, "ts": now + 0.001})

    # Persist to MongoDB
    try:
        if messages is not None:
            messages.insert_many(
                [
                    {"sid": sid, "role": "user", "content": user_text, "ts": now},
                    {"sid": sid, "role": "assistant", "content": bot_text, "ts": now + 0.001},
                ]
            )
    except Exception as e:
        print(f"[Store notice] Could not save exchange to MongoDB: {e}")


def history(sid, limit=20):
    # Try reading from MongoDB
    try:
        if messages is not None:
            rows = (
                messages.find({"sid": sid}, {"_id": 0, "role": 1, "content": 1})
                .sort("ts", DESCENDING)
                .limit(limit)
            )
            data = list(rows)[::-1]
            if data:
                return data
    except Exception as e:
        print(f"[Store notice] Error fetching from MongoDB ({e}), using memory.")

    # Fallback to local memory
    local = _local_messages.get(sid, [])
    return [{"role": m["role"], "content": m["content"]} for m in local[-limit:]]


def add_match(sid, title, score, verdict, matched, missing):
    item = {
        "sid": sid,
        "title": title,
        "score": score,
        "verdict": verdict,
        "matched": matched,
        "missing": missing,
        "ts": time.time(),
    }
    _local_matches[sid].append(item)

    try:
        if matches is not None:
            matches.insert_one(item)
    except Exception as e:
        print(f"[Store notice] Could not save match to MongoDB: {e}")


def recent_matches(sid, n=3):
    try:
        if matches is not None:
            res = list(matches.find({"sid": sid}, {"_id": 0}).sort("ts", DESCENDING).limit(n))
            if res:
                return res
    except Exception:
        pass

    local = _local_matches.get(sid, [])
    return sorted(local, key=lambda x: x.get("ts", 0), reverse=True)[:n]
