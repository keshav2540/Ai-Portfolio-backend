import json
import logging
import os
import re
import time
from typing import Generator, List, Optional, Set
import httpx
from dotenv import load_dotenv

load_dotenv()

# Task 1.2: Suppress verbose HTTP request logs to prevent URL/header leaks
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("llm")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Task 4.1: Timeout configuration
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
MAX_RETRIES = 3
BACKOFF_DELAYS = [1.0, 2.0, 4.0]

# Task 2.2 & 2.4: Dead models tracking in memory for this process
KNOWN_DEAD_MODELS: Set[str] = {
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-3.5-flash",
    "gemini-3.8-pro",
}
_dead_models: Set[str] = set()
_available_models_cache: Optional[Set[str]] = None


def _get_api_key() -> str:
    # Task 1.4: Only from environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""
    key = key.strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")
    return key


def _sanitize(text: str, key: str) -> str:
    """Task 1.3: Ensures the API key is never exposed in logs or exception messages."""
    if not text or not key:
        return text or ""
    return text.replace(key, "[REDACTED_API_KEY]")


def get_configured_models() -> List[str]:
    """Task 2.1: Primary and fallback models from environment variables."""
    primary = os.getenv("GEMINI_MODEL") or os.getenv("MODEL") or "gemini-3.8-flash"
    primary = primary.strip()

    fallbacks_raw = os.getenv("GEMINI_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]

    candidates = [primary] + fallbacks
    valid: List[str] = []
    for m in candidates:
        # Exclude models marked dead or known dead (unless user explicitly configured them as primary)
        if m in _dead_models:
            continue
        if m in KNOWN_DEAD_MODELS and m != primary:
            continue
        if m not in valid:
            valid.append(m)

    if not valid:
        # Fallback safeguard
        valid = [primary if primary not in _dead_models else "gemini-3.8-flash"]
    return valid


def _format_contents(messages: list[dict]) -> list[dict]:
    contents = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        content_text = m.get("content", "")
        if content_text:
            contents.append({
                "role": role,
                "parts": [{"text": content_text}],
            })
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    return contents


def ask(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
    deterministic: bool = False,
    json_mode: bool = False,
) -> str:
    key = _get_api_key()
    headers = {
        "x-goog-api-key": key,
        "Content-Type": "application/json",
    }

    payload = {"contents": _format_contents(messages)}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    gen_cfg = {}
    if max_tokens:
        gen_cfg["maxOutputTokens"] = max_tokens
    if deterministic:
        gen_cfg["temperature"] = 0.0
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    candidate_models = get_configured_models()
    last_error: Optional[Exception] = None

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        for model_name in candidate_models:
            if model_name in _dead_models:
                continue

            # Task 1.1: Gemini key in header, clean URL
            url = f"{BASE_URL}/models/{model_name}:generateContent"

            for attempt in range(MAX_RETRIES):
                try:
                    resp = client.post(url, headers=headers, json=payload)
                    status = resp.status_code

                    if status == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            return ""
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        return "".join(text_parts).strip()

                    body_text = _sanitize(resp.text, key)

                    # Task 2.4 & Task 4.3: 404 or no longer available -> mark dead, do not retry
                    if status == 404 or "no longer available" in body_text.lower() or "not found" in body_text.lower():
                        _dead_models.add(model_name)
                        logger.warning(
                            f"Model '{model_name}' returned {status} (unavailable). Marked dead. Trying next fallback."
                        )
                        last_error = RuntimeError(f"Model '{model_name}' not available.")
                        break  # Move to next fallback model immediately

                    # Task 4.2: 503, 429 -> retry same model with exponential backoff
                    if status in (503, 429):
                        logger.warning(
                            f"Gemini {model_name} returned status {status}. Attempt {attempt + 1}/{MAX_RETRIES}."
                        )
                        last_error = RuntimeError(f"AI service busy ({status}).")
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(BACKOFF_DELAYS[attempt])
                            continue
                        else:
                            break  # Retries exhausted for this model

                    # Other client errors (400, 401, 403)
                    logger.error(f"Gemini {model_name} returned unretryable status {status}: {body_text[:200]}")
                    last_error = RuntimeError(f"AI service error ({status}).")
                    break

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    logger.warning(
                        f"Network/Timeout with {model_name} on attempt {attempt + 1}/{MAX_RETRIES}: {type(exc).__name__}"
                    )
                    last_error = exc
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(BACKOFF_DELAYS[attempt])
                        continue
                    else:
                        break

    if last_error:
        raise last_error
    raise RuntimeError("AI service temporarily unavailable. Please try again later.")


def _clean_and_parse_json(raw: str) -> dict:
    """Helper to strip markdown fences, extract JSON and clean trailing commas."""
    raw = raw.strip()
    # 1. Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2. Extract outermost JSON object { ... }
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # 3. Clean trailing commas in object or array
            cleaned_commas = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(cleaned_commas)
            except Exception:
                pass

    raise ValueError(f"Could not parse valid JSON from AI response: {raw[:150]}")


def ask_json(system: str, user: str, max_tokens: int | None = None) -> dict:
    """Task 6.3: JSON query with fence stripping and 1 automatic retry on parse failure."""
    try:
        raw = ask(
            system,
            [{"role": "user", "content": user}],
            max_tokens,
            deterministic=True,
            json_mode=True,
        )
        return _clean_and_parse_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(f"ask_json parse failed on initial response: {e}. Retrying once with explicit JSON prompt...")
        retry_prompt = (
            f"{user}\n\nIMPORTANT: Return valid, RFC-compliant JSON ONLY. "
            f"Do not include explanation, markdown code fences, or any other characters."
        )
        raw = ask(
            system,
            [{"role": "user", "content": retry_prompt}],
            max_tokens,
            deterministic=True,
            json_mode=True,
        )
        return _clean_and_parse_json(raw)


def ask_stream(system: str, messages: list[dict], max_tokens: int | None = None) -> Generator[str, None, None]:
    """
    Streaming generator with:
    - Task 1.1: x-goog-api-key header
    - Task 3.1: resp.read() before reading response on non-200 (fixes ResponseNotRead)
    - Task 4.4: Does not retry if tokens were already emitted mid-stream
    - Task 4.2 & 4.3: 503/429 retries with backoff, 404 marks dead model
    """
    key = _get_api_key()
    headers = {
        "x-goog-api-key": key,
        "Content-Type": "application/json",
    }

    payload = {"contents": _format_contents(messages)}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    gen_cfg = {}
    if max_tokens:
        gen_cfg["maxOutputTokens"] = max_tokens
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    candidate_models = get_configured_models()
    tokens_yielded = 0
    last_error: Optional[Exception] = None

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        for model_name in candidate_models:
            if model_name in _dead_models:
                continue

            url = f"{BASE_URL}/models/{model_name}:streamGenerateContent?alt=sse"

            for attempt in range(MAX_RETRIES):
                try:
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        status = resp.status_code

                        if status == 200:
                            for line in resp.iter_lines():
                                if line.startswith("data: "):
                                    raw_chunk = line[6:].strip()
                                    if raw_chunk:
                                        try:
                                            chunk_data = json.loads(raw_chunk)
                                            candidates = chunk_data.get("candidates", [])
                                            if candidates:
                                                parts = candidates[0].get("content", {}).get("parts", [])
                                                for p in parts:
                                                    txt = p.get("text", "")
                                                    if txt:
                                                        tokens_yielded += 1
                                                        yield txt
                                        except Exception:
                                            continue
                            return

                        # Task 3.1: CRITICAL BUG FIX - Read response content before inspecting text!
                        resp.read()
                        body_text = _sanitize(resp.text, key)

                        # Task 2.4 & Task 4.3: 404 or no longer available
                        if status == 404 or "no longer available" in body_text.lower() or "not found" in body_text.lower():
                            _dead_models.add(model_name)
                            logger.warning(
                                f"Stream model '{model_name}' returned {status} (unavailable). Marked dead. Trying next fallback."
                            )
                            last_error = RuntimeError(f"Model '{model_name}' not available.")
                            break

                        # Task 4.2: 503, 429
                        if status in (503, 429):
                            logger.warning(
                                f"Stream Gemini {model_name} returned status {status}. Attempt {attempt + 1}/{MAX_RETRIES}."
                            )
                            last_error = RuntimeError(f"AI service busy ({status}).")
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(BACKOFF_DELAYS[attempt])
                                continue
                            else:
                                break

                        logger.error(f"Stream Gemini {model_name} returned unretryable status {status}: {body_text[:200]}")
                        last_error = RuntimeError(f"AI service error ({status}).")
                        break

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    # Task 4.4: Streaming me agar kuch tokens aa chuke hain, do not retry!
                    if tokens_yielded > 0:
                        logger.error(f"Network error mid-stream after emitting {tokens_yielded} tokens: {type(exc).__name__}")
                        raise RuntimeError("Streaming connection interrupted mid-stream.") from exc

                    logger.warning(
                        f"Stream Network/Timeout on {model_name} attempt {attempt + 1}/{MAX_RETRIES}: {type(exc).__name__}"
                    )
                    last_error = exc
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(BACKOFF_DELAYS[attempt])
                        continue
                    else:
                        break

    if last_error:
        raise last_error
    raise RuntimeError("AI service temporarily unavailable. Please try again later.")
