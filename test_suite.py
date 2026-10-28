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
from unittest.mock import patch

os.environ["TESTING"] = "1"
os.environ["GEMINI_API_KEY"] = "test-secret-gemini-key-12345"
os.environ["GEMINI_MODEL"] = "gemini-3.8-flash"
os.environ["GEMINI_FALLBACK_MODELS"] = "gemini-test-fallback"

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
from fastapi.testclient import TestClient

TEST_KEY = "test-secret-gemini-key-12345"


class TestLLMUnit(unittest.TestCase):
    def setUp(self):
        llm._dead_models.clear()
        llm.BACKOFF_DELAYS = [0.001, 0.001, 0.001]

    def tearDown(self):
        llm._dead_models.clear()

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
        with patch("httpx.Client", return_value=mock_client):
            res = llm.ask("system", [{"role": "user", "content": "hi"}])
            self.assertEqual(res, "Hello world")

        self.assertGreaterEqual(len(recorded_requests), 1)
        req = recorded_requests[0]
        self.assertEqual(req.headers.get("x-goog-api-key"), TEST_KEY)
        self.assertNotIn(TEST_KEY, str(req.url))
        self.assertNotIn("key=", str(req.url))

    def test_retry_on_503_and_succeed_on_third_attempt(self):
        """Task 4.2: 503 should retry same model and succeed on 3rd attempt."""
        calls = 0

        def handler(request: httpx.Request):
            nonlocal calls
            calls += 1
            if calls < 3:
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": "Success after retries"}]}}
                    ]
                },
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("httpx.Client", return_value=mock_client):
            res = llm.ask("system", [{"role": "user", "content": "test"}])
            self.assertEqual(res, "Success after retries")
            self.assertEqual(calls, 3)

    def test_404_marks_model_dead_and_switches_to_fallback(self):
        """Task 2.4 & 4.3: 404 should mark model dead and switch to next model without retrying 404."""
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
        with patch("httpx.Client", return_value=mock_client):
            res = llm.ask("system", [{"role": "user", "content": "test"}])
            self.assertEqual(res, "Fallback success")

        self.assertIn("gemini-3.8-flash", llm._dead_models)
        flash_calls = [p for p in requested_models if "gemini-3.8-flash" in p]
        self.assertEqual(len(flash_calls), 1)

    def test_stream_non_200_does_not_raise_response_not_read(self):
        """Task 3.1: Streaming error must not raise httpx.ResponseNotRead."""
        def handler(request: httpx.Request):
            return httpx.Response(503, text="Service Unavailable upstream")

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("httpx.Client", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                list(llm.ask_stream("system", [{"role": "user", "content": "test"}]))
            self.assertNotIn("ResponseNotRead", str(ctx.exception))

    def test_all_models_fail_raises_clean_runtime_error(self):
        """Task 4: When all models fail, clean error is raised."""
        def handler(request: httpx.Request):
            return httpx.Response(503, text="Service Unavailable")

        mock_client = httpx.Client(transport=httpx.MockTransport(handler))
        with patch("httpx.Client", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                llm.ask("system", [{"role": "user", "content": "test"}])
            self.assertIn("AI service", str(ctx.exception))

    def test_api_key_not_in_error_logs(self):
        """Task 1.3: API key must never appear in log output."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger("llm")
        logger.addHandler(handler)

        def mock_error_handler(request: httpx.Request):
            return httpx.Response(400, text=f"Invalid request with {TEST_KEY}")

        mock_client = httpx.Client(transport=httpx.MockTransport(mock_error_handler))
        with patch("httpx.Client", return_value=mock_client):
            try:
                llm.ask("system", [{"role": "user", "content": "hi"}])
            except Exception:
                pass

        logger.removeHandler(handler)
        log_output = log_stream.getvalue()
        self.assertNotIn(TEST_KEY, log_output)


class TestFastAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_head_root_returns_200(self):
        """Task 8: HEAD / must return 200 OK for Render health checks."""
        res = self.client.head("/")
        self.assertEqual(res.status_code, 200)

    def test_get_root_returns_200(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "online")

    def test_match_returns_503_on_upstream_failure(self):
        """Task 6: /api/match returns 503 (not 502) on AI timeout/failure."""
        with patch("main.rag.get_all_indexed_resume", return_value="Resume content"), \
             patch("main.matcher.run", side_effect=RuntimeError("AI service temporarily unavailable (503)")):
            res = self.client.post(
                "/api/match",
                data={"session_id": "test-session-12345", "jd_text": "Looking for React developer with 3 years exp" * 5},
            )
            # Must NOT be 502
            self.assertNotEqual(res.status_code, 502)
            self.assertEqual(res.status_code, 503)
            self.assertIn("Abhi AI service busy hai", res.json().get("detail", ""))

    def test_chat_non_stream_returns_503_on_upstream_failure(self):
        """Task 6: /api/chat returns 503 on AI failure."""
        with patch("main.get_resume_context", return_value="Resume content"), \
             patch("main.ask", side_effect=RuntimeError("AI service temporarily unavailable (503)")):
            res = self.client.post(
                "/api/chat",
                json={"session_id": "test-session-12345", "message": "hello"},
            )
            self.assertEqual(res.status_code, 503)
            self.assertIn("Abhi AI service busy hai", res.json().get("detail", ""))

    def test_history_normal_flow(self):
        """Verify /api/history endpoint functions normally."""
        res = self.client.get("/api/history?session_id=test-session-12345")
        self.assertEqual(res.status_code, 200)
        self.assertIn("messages", res.json())

    def test_stream_generator_catches_error_and_yields_friendly_message(self):
        """Task 5: Verify the streaming generator handles errors and yields friendly message."""
        # Directly test chat_stream response generator logic
        def failing_ask_stream(system, msgs):
            raise RuntimeError("503 Service Unavailable")
            yield "never"

        with patch("main.ask_stream", side_effect=failing_ask_stream):
            # Instantiate request and body
            body = main.ChatIn(session_id="test-session-12345", message="test")
            dummy_request = unittest.mock.MagicMock()
            dummy_request.client.host = "127.0.0.1"

            resp = main.chat_stream(body, dummy_request)
            chunks = list(resp.body_iterator)
            full_text = "".join(chunks)
            self.assertIn("Abhi AI service busy hai", full_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
