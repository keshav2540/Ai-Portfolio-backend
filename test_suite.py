"""
Production Test Suite for AI Portfolio Backend
Run with: python test_suite.py
"""
import io
import json
import logging
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import httpx

os.environ["TESTING"] = "1"
os.environ["GEMINI_API_KEY"] = "test-secret-gemini-key-12345"
os.environ["GEMINI_MODEL"] = "gemini-3.8-flash"
os.environ["GEMINI_FALLBACK_MODELS"] = "gemini-test-fallback"
os.environ["GEMINI_THINKING_LEVEL"] = "low"
os.environ["LLM_READ_TIMEOUT"] = "30"
os.environ["FIRST_TOKEN_TIMEOUT"] = "25"

sys.path.insert(0, os.path.dirname(__file__))

# Suppress logs during tests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import store
store.messages = None
store.matches = None
store.init = lambda: None

import llm
import main

TEST_KEY = "test-secret-gemini-key-12345"


class TestLLMUnit(unittest.TestCase):
    def setUp(self):
        llm._dead_models.clear()
        llm.BACKOFF_DELAYS = [0.001, 0.001, 0.001]
        llm._response_cache.clear()  # clear cache so tests don't bleed into each other
        llm._groq_client = None      # reset Groq singleton

    def tearDown(self):
        llm._dead_models.clear()
        llm._response_cache.clear()

    def test_api_key_sent_in_header_not_in_url(self):
        """Task 1.1: Verify API key is passed via x-goog-api-key header and NOT in URL query."""
        recorded_requests = []

        def handler(request: httpx.Request):
            recorded_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Hello world"}]}}
                    ]
                },
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            res = llm.ask("system", [{"role": "user", "content": "hi"}])
            self.assertEqual(res, "Hello world")

        self.assertGreaterEqual(len(recorded_requests), 1)
        req = recorded_requests[0]
        self.assertEqual(req.headers.get("x-goog-api-key"), TEST_KEY)
        self.assertNotIn(TEST_KEY, str(req.url))
        self.assertNotIn("key=", str(req.url))

    def test_thinking_config_injected_in_generation_config(self):
        """Task 2: Verify thinkingConfig with thinkingLevel is injected for Gemini 3 models."""
        recorded_requests = []

        def handler(request: httpx.Request):
            recorded_requests.append(request)
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Answer"}]}}
                    ]
                },
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            llm.ask("system", [{"role": "user", "content": "hi"}])

        self.assertGreaterEqual(len(recorded_requests), 1)
        body = json.loads(recorded_requests[0].content.decode("utf-8"))
        self.assertIn("generationConfig", body)
        self.assertIn("thinkingConfig", body["generationConfig"])
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low")

    def test_retry_on_read_timeout_and_succeed_on_second_attempt(self):
        """Task 3: ReadTimeout on 1st attempt retries same model and succeeds on 2nd attempt."""
        attempts = 0

        def handler(request: httpx.Request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("The read operation timed out")
            # 2nd attempt succeeds with streaming SSE format
            sse_content = 'data: {"candidates": [{"content": {"parts": [{"text": "Stream success"}]}}]}\n\n'
            return httpx.Response(
                200,
                content=sse_content.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            result = list(llm.ask_stream("system", [{"role": "user", "content": "test"}]))
            self.assertEqual(result, ["Stream success"])
            self.assertEqual(attempts, 2)

    def test_first_token_timeout_triggers_retry(self):
        """Task 1 & 3: First token timeout triggers smart retry on the same model."""
        attempts = 0

        def handler(request: httpx.Request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ReadTimeout("First token read timed out after 25s")
            sse_content = 'data: {"candidates": [{"content": {"parts": [{"text": "Token arrived"}]}}]}\n\n'
            return httpx.Response(
                200,
                content=sse_content.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            chunks = list(llm.ask_stream("system", [{"role": "user", "content": "test"}]))
            self.assertEqual(chunks, ["Token arrived"])
            self.assertEqual(attempts, 2)

    def test_404_marks_model_dead_and_switches_to_fallback(self):
        """Task 2.4 & 4.3: 404 should mark model dead and switch to next model immediately without retries."""
        requested_models = []

        def handler(request: httpx.Request):
            requested_models.append(request.url.path)
            if "gemini-3.8-flash" in request.url.path:
                return httpx.Response(404, text='{"error": "models/gemini-3.8-flash is not found"}')
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Fallback success"}]}}
                    ]
                },
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            res = llm.ask("system", [{"role": "user", "content": "test"}])
            self.assertEqual(res, "Fallback success")

        self.assertIn("gemini-3.8-flash", llm._dead_models)
        flash_calls = [p for p in requested_models if "gemini-3.8-flash" in p]
        self.assertEqual(len(flash_calls), 1)

    def test_mid_stream_network_error_does_not_retry(self):
        """Task 3: If tokens were already emitted mid-stream, do NOT retry (avoid duplicate output)."""
        calls = 0

        class MockStreamTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                def chunk_generator():
                    yield b'data: {"candidates": [{"content": {"parts": [{"text": "Part1"}]}}]}\n\n'
                    raise httpx.ReadTimeout("Connection dropped mid-stream")
                return httpx.Response(200, content=chunk_generator())

        mock_client = httpx.Client(transport=MockStreamTransport())
        with patch("llm.get_client", return_value=mock_client):
            gen = llm.ask_stream("system", [{"role": "user", "content": "test"}])
            first_chunk = next(gen)
            self.assertEqual(first_chunk, "Part1")
            with self.assertRaises(RuntimeError) as ctx:
                next(gen)
            self.assertIn("interrupted", str(ctx.exception))
            self.assertEqual(calls, 1)

    def test_all_streaming_attempts_fail_falls_back_to_non_streaming(self):
        """Task 4: When all streaming attempts fail, fall back to non-streaming generateContent."""
        stream_attempts = 0

        def handler(request: httpx.Request):
            nonlocal stream_attempts
            if "streamGenerateContent" in str(request.url):
                stream_attempts += 1
                raise httpx.ReadTimeout("Stream read timeout on attempt")
            else:
                return httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {"content": {"parts": [{"text": "Non-streaming fallback reply"}]}}
                        ]
                    },
                )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            chunks = list(llm.ask_stream("system", [{"role": "user", "content": "test"}]))
            self.assertEqual(chunks, ["Non-streaming fallback reply"])
            self.assertGreaterEqual(stream_attempts, 2)

    def test_stream_non_200_does_not_raise_response_not_read(self):
        """Task 3.1: Streaming error must not raise httpx.ResponseNotRead."""
        def handler(request: httpx.Request):
            return httpx.Response(503, text="Service Unavailable upstream")

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("llm.get_client", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                list(llm.ask_stream("system", [{"role": "user", "content": "test"}]))
            self.assertNotIn("ResponseNotRead", str(ctx.exception))

    def test_api_key_not_in_error_logs(self):
        """Task 1.3: API key must never appear in log output."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("llm")
        logger.addHandler(handler)

        def mock_error_handler(request: httpx.Request):
            return httpx.Response(400, text=f"Invalid request with {TEST_KEY}")

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_error_handler))
        with patch("llm.get_client", return_value=mock_client):
            try:
                llm.ask("system", [{"role": "user", "content": "hi"}])
            except Exception:
                pass

        logger.removeHandler(handler)
        log_output = log_stream.getvalue()
        self.assertNotIn(TEST_KEY, log_output)


class TestFastAPIIntegration(unittest.TestCase):
    def test_root_endpoint_returns_online(self):
        """Task 8: root route returns online status and health check."""
        res = main.root()
        self.assertEqual(res["status"], "online")
        self.assertIn("health", res)

    def test_match_returns_503_on_upstream_failure(self):
        """Task 6: match endpoint returns 503 (not 502) on AI timeout/failure."""
        dummy_req = MagicMock()
        dummy_req.client.host = "127.0.0.1"
        with patch("main.rag.get_all_indexed_resume", return_value="Resume content"), \
             patch("main.matcher.run", side_effect=RuntimeError("AI service temporarily unavailable (503)")):
            with self.assertRaises(main.HTTPException) as ctx:
                main.match(dummy_req, session_id="test-session-12345", file=None, jd_text="Looking for React developer with 3 years exp" * 5)
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("Abhi AI service busy hai", ctx.exception.detail)

    def test_chat_non_stream_returns_503_on_upstream_failure(self):
        """Task 6: /api/chat returns 503 on AI failure."""
        dummy_req = MagicMock()
        dummy_req.client.host = "127.0.0.1"
        body = main.ChatIn(session_id="test-session-12345", message="hello")
        with patch("main.get_resume_context", return_value="Resume content"), \
             patch("main.ask", side_effect=RuntimeError("AI service temporarily unavailable (503)")):
            with self.assertRaises(main.HTTPException) as ctx:
                main.chat(body, dummy_req)
            self.assertEqual(ctx.exception.status_code, 503)
            self.assertIn("Abhi AI service busy hai", ctx.exception.detail)

    def test_stream_generator_catches_error_and_yields_friendly_message(self):
        """Task 5 & 6: Verify the streaming generator handles errors and yields friendly message."""
        with patch("main.ask_stream", side_effect=RuntimeError("503 Service Unavailable")), \
             patch("main.get_resume_context", return_value="Mock resume"):
            body = main.ChatIn(session_id="test-session-12345", message="test")
            dummy_request = MagicMock()
            dummy_request.client.host = "127.0.0.1"

            resp = main.chat_stream(body, dummy_request)
            # Access underlying generator from StreamingResponse
            sync_gen = resp.body_iterator.ag_frame.f_locals.get("iterator", resp.body_iterator)
            chunks = list(sync_gen)
            full_text = "".join(chunks)
            self.assertIn("Abhi AI service busy hai", full_text)


class TestGroqIntegration(unittest.TestCase):
    """Tests for Groq LLM provider integration in llm.py."""

    def setUp(self):
        llm._dead_models.clear()
        llm._groq_client = None       # reset singleton before each test
        llm._response_cache.clear()   # clear cache

    def tearDown(self):
        llm._dead_models.clear()
        llm._groq_client = None
        llm._response_cache.clear()

    # ------------------------------------------------------------------
    # 1. Groq stream success: yields plain text chunks (format unchanged)
    # ------------------------------------------------------------------
    def test_groq_stream_success_format(self):
        """Groq stream yields plain text strings, same format as Gemini."""
        from langchain_core.messages import AIMessageChunk

        def fake_stream(messages, **kwargs):
            for word in ["Hello", " ", "world"]:
                yield AIMessageChunk(content=word)

        mock_groq = MagicMock()
        mock_groq.stream.side_effect = fake_stream

        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key", "GROQ_MODEL": "llama-3.3-70b-versatile"}):
            chunks = list(llm.ask_stream("system", [{"role": "user", "content": "hi"}]))

        self.assertEqual(chunks, ["Hello", " ", "world"])

    # ------------------------------------------------------------------
    # 2. Groq ask() failure falls back to Gemini
    # ------------------------------------------------------------------
    def test_groq_failure_falls_back_to_gemini(self):
        """If Groq.invoke() raises, ask() falls back to Gemini httpx path."""
        mock_groq = MagicMock()
        mock_groq.invoke.side_effect = RuntimeError("Groq 429 rate limit")

        def gemini_handler(request: httpx.Request):
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Gemini fallback"}]}}]},
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(gemini_handler))
        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch("llm.get_client", return_value=mock_client), \
             patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            result = llm.ask("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result, "Gemini fallback")
        mock_groq.invoke.assert_called_once()

    # ------------------------------------------------------------------
    # 3. Groq mid-stream error does not switch to Gemini
    # ------------------------------------------------------------------
    def test_groq_mid_stream_error_no_provider_switch(self):
        """After tokens are already yielded, mid-stream Groq error raises RuntimeError; Gemini NOT called."""
        from langchain_core.messages import AIMessageChunk

        def fake_stream_with_error(messages, **kwargs):
            yield AIMessageChunk(content="First token")
            raise RuntimeError("Connection reset mid-stream")

        mock_groq = MagicMock()
        mock_groq.stream.side_effect = fake_stream_with_error

        gemini_called = []

        def gemini_handler(request: httpx.Request):
            gemini_called.append(True)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "Gemini"}]}}]})

        mock_client = httpx.Client(transport=httpx.MockTransport(gemini_handler))
        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch("llm.get_client", return_value=mock_client), \
             patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            gen = llm.ask_stream("system", [{"role": "user", "content": "test"}])
            first = next(gen)
            self.assertEqual(first, "First token")
            with self.assertRaises(RuntimeError) as ctx:
                next(gen)

        self.assertIn("interrupted", str(ctx.exception).lower())
        self.assertEqual(gemini_called, [], "Gemini must NOT be called after mid-stream Groq error")

    # ------------------------------------------------------------------
    # 4. ask_json() strips markdown fences from Groq output
    # ------------------------------------------------------------------
    def test_groq_json_fences_stripped_in_ask_json(self):
        """ask_json() correctly strips markdown fences from Groq response."""
        mock_groq = MagicMock()
        mock_groq.invoke.return_value = MagicMock(content='```json\n{"score": 85}\n```')

        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            result = llm.ask_json("system", "rate the resume")

        self.assertEqual(result, {"score": 85})

    # ------------------------------------------------------------------
    # 5. GROQ_API_KEY not set → Groq skipped, no crash
    # ------------------------------------------------------------------
    def test_groq_key_not_set_uses_gemini(self):
        """If GROQ_API_KEY is not set, Groq is silently skipped and Gemini is used."""
        def gemini_handler(request: httpx.Request):
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Gemini only"}]}}]},
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(gemini_handler))
        env_without_groq = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch("llm.get_client", return_value=mock_client), \
             patch.dict(os.environ, env_without_groq, clear=True):
            result = llm.ask("system", [{"role": "user", "content": "hi"}])

        self.assertEqual(result, "Gemini only")

    # ------------------------------------------------------------------
    # 6. GROQ_API_KEY never appears in logs
    # ------------------------------------------------------------------
    def test_groq_key_not_in_logs(self):
        """GROQ_API_KEY value must never appear in log output."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        llm_logger = logging.getLogger("llm")
        llm_logger.addHandler(handler)

        fake_groq_key = "gsk_test_secret_groq_key_xyz"
        mock_groq = MagicMock()
        mock_groq.invoke.side_effect = RuntimeError(f"Error with key {fake_groq_key}")

        def gemini_handler(request: httpx.Request):
            return httpx.Response(400, text="bad request")

        mock_client = httpx.Client(transport=httpx.MockTransport(gemini_handler))
        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch("llm.get_client", return_value=mock_client), \
             patch.dict(os.environ, {"GROQ_API_KEY": fake_groq_key}):
            try:
                llm.ask("system", [{"role": "user", "content": "hi"}])
            except Exception:
                pass

        llm_logger.removeHandler(handler)
        log_output = log_stream.getvalue()
        self.assertNotIn(fake_groq_key, log_output, "GROQ_API_KEY must not appear in logs")

    # ------------------------------------------------------------------
    # 7. Same question is cached — Groq called only once
    # ------------------------------------------------------------------
    def test_groq_response_cached(self):
        """Second identical ask() call uses cache, Groq invoked only once."""
        mock_groq = MagicMock()
        mock_groq.invoke.return_value = MagicMock(content="Cached answer")

        with patch("llm._get_groq_client", return_value=mock_groq), \
             patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"}):
            r1 = llm.ask("system", [{"role": "user", "content": "same question"}])
            r2 = llm.ask("system", [{"role": "user", "content": "same question"}])

        self.assertEqual(r1, "Cached answer")
        self.assertEqual(r2, "Cached answer")
        self.assertEqual(mock_groq.invoke.call_count, 1, "Groq should only be called once (second uses cache)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
