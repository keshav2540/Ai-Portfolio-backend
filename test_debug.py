import os
os.environ["TESTING"] = "1"
os.environ["GEMINI_API_KEY"] = "test-secret-gemini-key-12345"
os.environ["GEMINI_MODEL"] = "gemini-3.8-flash"
os.environ["GEMINI_FALLBACK_MODELS"] = "gemini-test-fallback"

import asyncio
from unittest.mock import patch, MagicMock

import store
store.messages = None
store.matches = None
store.init = lambda: None

import main

print("Step 1: main imported")

def failing_ask_stream(system, msgs, max_tokens=None):
    print("Step 3: failing_ask_stream called, raising error")
    raise RuntimeError("503 Service Unavailable")
    yield "never"

with patch("main.ask_stream", side_effect=failing_ask_stream), \
     patch("main.get_resume_context", return_value="Mock resume"):
    print("Step 2: inside with patch")
    body = main.ChatIn(session_id="test-session-12345", message="test")
    dummy_request = MagicMock()
    dummy_request.client.host = "127.0.0.1"

    resp = main.chat_stream(body, dummy_request)
    print("Step 4: chat_stream returned StreamingResponse")

    async def drain():
        print("Step 5: inside drain, iterating body_iterator")
        chunks = []
        async for chunk in resp.body_iterator:
            print("Step 6: got chunk:", repr(chunk))
            chunks.append(chunk)
        print("Step 7: finished iterating")
        return chunks

    result = asyncio.run(drain())
    print("Step 8: drain completed, result:", result)

