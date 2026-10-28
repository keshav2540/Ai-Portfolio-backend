import json
import logging
import os
import random
import re
import time
from typing import Any, Generator, List, Optional, Set
import httpx
from dotenv import load_dotenv

load_dotenv()

# Task 1.2: Suppress verbose HTTP request logs to prevent URL/header leaks
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Suppress LangChain / Groq SDK loggers to prevent key/URL leaks
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_groq").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.WARNING)

logger = logging.getLogger("llm")

# ---------------------------------------------------------------------------
# Groq / LangChain imports — optional; Gemini path still works without them
# ---------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Groq singleton and 10-min in-memory response cache
# ---------------------------------------------------------------------------
_groq_client: Optional[Any] = None
_response_cache: dict = {}   # {cache_key: (response_text, timestamp)}
_CACHE_TTL = 600             # 10 minutes

MAX_TOTAL_CALLS = 4          # Budget per user request across Groq + Gemini

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Task 3 & 4: Retry configuration (initial + 2 retries = 3 attempts total)
MAX_RETRIES = 3
BACKOFF_DELAYS = [1.0, 2.0]

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

# Task 7: Singleton reusable HTTP client with connection pooling
_client_instance: Optional[httpx.Client] = None


def get_client() -> httpx.Client:
    global _client_instance
    if os.getenv("TESTING") == "1":
        # In testing mode, create a client directly so mock transports/patches apply cleanly
        return httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        )
    if _client_instance is None or _client_instance.is_closed:
        _client_instance = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _client_instance


def _get_timeouts():
    """Task 1: Differentiated timeouts for streaming, first token, and non-streaming."""
    try:
        read_timeout = float(os.getenv("LLM_READ_TIMEOUT", "30.0"))
    except ValueError:
        read_timeout = 30.0

    try:
        first_token_timeout = float(os.getenv("FIRST_TOKEN_TIMEOUT", "25.0"))
    except ValueError:
        first_token_timeout = 25.0

    timeout_streaming_first = httpx.Timeout(connect=10.0, read=first_token_timeout, write=10.0, pool=10.0)
    timeout_streaming_packet = httpx.Timeout(connect=10.0, read=read_timeout, write=10.0, pool=10.0)
    timeout_non_streaming = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    return timeout_streaming_first, timeout_streaming_packet, timeout_non_streaming


def _get_thinking_config(model_name: str) -> Optional[dict]:
    """
    Task 2: Verified Gemini REST API format for reasoning models (e.g. Gemini 3 series).
    Uses generationConfig.thinkingConfig.thinkingLevel ('minimal', 'low', 'medium', 'high').
    Default is 'low' to minimize latency and prevent ReadTimeout.
    """
    level = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
    if not level or level in ("none", "off", "0"):
        return None
    # Only send thinkingConfig if model is Gemini 3 series or supports it
    if "gemini-3" in model_name and level in ("minimal", "low", "medium", "high"):
        return {"thinkingLevel": level}
    return None


def _get_api_key() -> str:
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
    """Task 2.1 & 5: Primary and fallback models from environment variables."""
    primary = os.getenv("GEMINI_MODEL") or os.getenv("MODEL") or "gemini-3.8-flash"
    primary = primary.strip()

    fallbacks_raw = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash-lite")
    fallbacks = [m.strip() for m in fallbacks_raw.split(",") if m.strip()]

    candidates = [primary] + fallbacks
    valid: List[str] = []
    for m in candidates:
        if m in _dead_models:
            continue
        if m in KNOWN_DEAD_MODELS and m != primary:
            continue
        if m not in valid:
            valid.append(m)

    if not valid:
        valid = [primary if primary not in _dead_models else "gemini-3.8-flash"]
    return valid


def check_available_models(key: str) -> Set[str]:
    """Task 5: Dynamically fetch valid models supporting generateContent from Gemini API."""
    global _available_models_cache
    if _available_models_cache is not None:
        return _available_models_cache
    try:
        client = get_client()
        headers = {"x-goog-api-key": key}
        resp = client.get(
            f"{BASE_URL}/models",
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
        )
        if resp.status_code == 200:
            data = resp.json()
            models = set()
            for item in data.get("models", []):
                name = item.get("name", "").replace("models/", "")
                methods = item.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    models.add(name)
            _available_models_cache = models
            return models
    except Exception as e:
        logger.warning(f"Could not dynamically query Gemini models ({type(e).__name__}); using configured fallback list.")
    return set()


# ---------------------------------------------------------------------------
# Groq helper functions
# ---------------------------------------------------------------------------

def _cache_key(system: str, messages: list) -> str:
    import hashlib
    raw = system + json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    entry = _response_cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


def _cache_set(key: str, value: str) -> None:
    _response_cache[key] = (value, time.time())


def _build_lc_messages(system: str, messages: list) -> list:
    """Convert dicts to LangChain message objects."""
    lc = []
    if system:
        lc.append(SystemMessage(content=system))
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    if not lc or not any(isinstance(x, HumanMessage) for x in lc):
        lc.append(HumanMessage(content="Hello"))
    return lc


def _get_groq_client() -> Any:
    """Singleton ChatGroq instance. Never logs the API key."""
    global _groq_client
    if _groq_client is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        try:
            temperature = float(os.getenv("GROQ_TEMPERATURE", "0.7"))
        except ValueError:
            temperature = 0.7
        try:
            timeout = float(os.getenv("GROQ_TIMEOUT", "30.0"))
        except ValueError:
            timeout = 30.0
        try:
            max_retries = int(os.getenv("GROQ_MAX_RETRIES", "1"))
        except ValueError:
            max_retries = 1
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        _groq_client = ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            api_key=api_key,
        )
    return _groq_client


def _validate_groq_model() -> None:
    """Called at startup: logs a warning if GROQ_API_KEY or GROQ_MODEL is missing. Never crashes."""
    if not _LANGCHAIN_AVAILABLE:
        logger.info("langchain-groq not installed; Groq provider disabled, Gemini will be used.")
        return
    if not os.getenv("GROQ_API_KEY", "").strip():
        logger.info("GROQ_API_KEY not set; Groq provider disabled, Gemini will be used.")
        return
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    try:
        _get_groq_client()
        logger.info(f"Groq provider enabled with model '{model}'.")
    except Exception as e:
        logger.warning(f"Groq provider init failed ({type(e).__name__}); Gemini fallback will be used.")


# Run validation at module load
try:
    _validate_groq_model()
except Exception:
    pass


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
    # ------------------------------------------------------------------
    # 1. Cache lookup (only for non-json_mode to keep it safe)
    # ------------------------------------------------------------------
    if not json_mode:
        ck = _cache_key(system, messages)
        cached = _cache_get(ck)
        if cached is not None:
            logger.info("ask(): Returning cached response.")
            return cached

    total_calls = 0
    last_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # 2. Try Groq first (if configured and budget allows)
    # ------------------------------------------------------------------
    if _LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY", "").strip() and total_calls < MAX_TOTAL_CALLS:
        try:
            groq = _get_groq_client()
            lc_messages = _build_lc_messages(system, messages)
            start = time.time()
            response = groq.invoke(lc_messages)
            elapsed = time.time() - start
            total_calls += 1
            model_name = os.getenv("GROQ_MODEL", "groq")
            text = (response.content or "").strip()
            logger.info(f"Groq '{model_name}' success in {elapsed:.2f}s (call {total_calls}/{MAX_TOTAL_CALLS})")
            if not json_mode:
                _cache_set(ck, text)
            return text
        except Exception as groq_exc:
            total_calls += 1
            logger.warning(f"Groq ask() failed ({type(groq_exc).__name__}); falling back to Gemini.")
            last_error = groq_exc

    if total_calls >= MAX_TOTAL_CALLS:
        if last_error:
            raise last_error
        raise RuntimeError("AI service temporarily unavailable. Please try again later.")

    # ------------------------------------------------------------------
    # 3. Gemini httpx path (unchanged)
    # ------------------------------------------------------------------
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

    _, _, timeout_non_streaming = _get_timeouts()
    candidate_models = get_configured_models()
    client = get_client()

    for model_name in candidate_models:
        if model_name in _dead_models:
            continue

        # Task 2: Inject thinkingConfig for models supporting reasoning
        thinking_cfg = _get_thinking_config(model_name)
        if thinking_cfg:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["thinkingConfig"] = thinking_cfg
        elif "generationConfig" in payload and "thinkingConfig" in payload["generationConfig"]:
            del payload["generationConfig"]["thinkingConfig"]

        # Clean URL, key passed via x-goog-api-key header
        url = f"{BASE_URL}/models/{model_name}:generateContent"

        for attempt in range(MAX_RETRIES):
            start_time = time.time()
            try:
                resp = client.post(url, headers=headers, json=payload, timeout=timeout_non_streaming)
                elapsed = time.time() - start_time
                status = resp.status_code

                if status == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    result_text = "".join(text_parts).strip()
                    if not json_mode:
                        _cache_set(_cache_key(system, messages), result_text)
                    return result_text

                body_text = _sanitize(resp.text, key)

                # Task 2.4 & Task 4.3: 404 or no longer available -> mark dead, do not retry
                if status == 404 or "no longer available" in body_text.lower() or "not found" in body_text.lower():
                    _dead_models.add(model_name)
                    logger.warning(
                        f"Model '{model_name}' returned {status} in {elapsed:.2f}s. Marked dead. Trying next fallback model."
                    )
                    last_error = RuntimeError(f"Model '{model_name}' not available.")
                    break

                # Task 3: 503, 429 -> retry same model with exponential backoff + jitter
                if status in (503, 429):
                    last_error = RuntimeError(f"AI service busy ({status}).")
                    if attempt < MAX_RETRIES - 1:
                        base_delay = BACKOFF_DELAYS[attempt] if attempt < len(BACKOFF_DELAYS) else 2.0
                        delay = base_delay + random.uniform(0.1, 0.4)
                        logger.warning(
                            f"Gemini '{model_name}' returned status {status} in {elapsed:.2f}s (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.warning(
                            f"Retries exhausted for model '{model_name}' (status {status}) in {elapsed:.2f}s. Trying next fallback model."
                        )
                        break

                logger.error(f"Gemini '{model_name}' returned unretryable status {status} in {elapsed:.2f}s: {body_text[:200]}")
                last_error = RuntimeError(f"AI service error ({status}).")
                break

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                elapsed = time.time() - start_time
                reason = type(exc).__name__
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    base_delay = BACKOFF_DELAYS[attempt] if attempt < len(BACKOFF_DELAYS) else 2.0
                    delay = base_delay + random.uniform(0.1, 0.4)
                    logger.warning(
                        f"Network/Timeout with '{model_name}' on attempt {attempt + 1}/{MAX_RETRIES} ({reason}) in {elapsed:.2f}s. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Network retries exhausted for '{model_name}' ({reason}) in {elapsed:.2f}s. Trying next fallback model."
                    )
                    break

    if last_error:
        raise last_error
    raise RuntimeError("AI service temporarily unavailable. Please try again later.")


def _clean_and_parse_json(raw: str) -> dict:
    """Helper to strip markdown fences, extract JSON and clean trailing commas."""
    raw = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
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
    - Groq primary: streams tokens via ChatGroq.stream()
    - Task 1: Separate FIRST_TOKEN_TIMEOUT and LLM_READ_TIMEOUT
    - Task 2: Thinking overhead reduction via thinkingLevel: 'low'
    - Task 3: Smart retry (ReadTimeout, 503, 429) with backoff + jitter; 404 marks dead
    - Task 4: Fallback to non-streaming generateContent if all streaming attempts fail
    - Task 7: Reusable HTTP connection
    """
    tokens_yielded = 0
    total_calls = 0

    # ------------------------------------------------------------------
    # 1. Try Groq streaming first
    # ------------------------------------------------------------------
    if _LANGCHAIN_AVAILABLE and os.getenv("GROQ_API_KEY", "").strip() and total_calls < MAX_TOTAL_CALLS:
        try:
            groq = _get_groq_client()
            lc_messages = _build_lc_messages(system, messages)
            start = time.time()
            for chunk in groq.stream(lc_messages):
                txt = chunk.content if hasattr(chunk, "content") else ""
                if txt:
                    tokens_yielded += 1
                    yield txt
            total_calls += 1
            model_name = os.getenv("GROQ_MODEL", "groq")
            logger.info(f"Groq '{model_name}' stream complete, {tokens_yielded} tokens in {time.time()-start:.2f}s (call {total_calls}/{MAX_TOTAL_CALLS})")
            return
        except Exception as groq_exc:
            total_calls += 1
            if tokens_yielded > 0:
                # Mid-stream error — do NOT switch providers (would duplicate text)
                logger.error(f"Groq mid-stream error after {tokens_yielded} tokens ({type(groq_exc).__name__}); cannot switch mid-stream.")
                raise RuntimeError("Streaming connection interrupted mid-stream.") from groq_exc
            logger.warning(f"Groq stream failed ({type(groq_exc).__name__}); falling back to Gemini.")

    # ------------------------------------------------------------------
    # 2. Gemini streaming path (unchanged)
    # ------------------------------------------------------------------
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

    timeout_streaming_first, _, _ = _get_timeouts()
    candidate_models = get_configured_models()
    last_error: Optional[Exception] = None
    client = get_client()

    for model_name in candidate_models:
        if model_name in _dead_models:
            continue

        # Task 2: Inject thinkingConfig for models supporting reasoning
        thinking_cfg = _get_thinking_config(model_name)
        if thinking_cfg:
            if "generationConfig" not in payload:
                payload["generationConfig"] = {}
            payload["generationConfig"]["thinkingConfig"] = thinking_cfg
        elif "generationConfig" in payload and "thinkingConfig" in payload["generationConfig"]:
            del payload["generationConfig"]["thinkingConfig"]

        url = f"{BASE_URL}/models/{model_name}:streamGenerateContent?alt=sse"

        for attempt in range(MAX_RETRIES):
            start_time = time.time()
            try:
                with client.stream("POST", url, headers=headers, json=payload, timeout=timeout_streaming_first) as resp:
                    elapsed = time.time() - start_time
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

                    # Task 3.1: Read content on non-200 to prevent ResponseNotRead
                    resp.read()
                    body_text = _sanitize(resp.text, key)

                    # 404 or no longer available -> mark dead, do not retry
                    if status == 404 or "no longer available" in body_text.lower() or "not found" in body_text.lower():
                        _dead_models.add(model_name)
                        logger.warning(
                            f"Stream model '{model_name}' returned {status} in {elapsed:.2f}s. Marked dead. Trying next fallback model."
                        )
                        last_error = RuntimeError(f"Model '{model_name}' not available.")
                        break

                    # 503, 429 -> retry same model
                    if status in (503, 429):
                        last_error = RuntimeError(f"AI service busy ({status}).")
                        if attempt < MAX_RETRIES - 1:
                            base_delay = BACKOFF_DELAYS[attempt] if attempt < len(BACKOFF_DELAYS) else 2.0
                            delay = base_delay + random.uniform(0.1, 0.4)
                            logger.warning(
                                f"Stream Gemini '{model_name}' returned status {status} in {elapsed:.2f}s (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {delay:.2f}s..."
                            )
                            time.sleep(delay)
                            continue
                        else:
                            logger.warning(
                                f"Stream retries exhausted for '{model_name}' (status {status}) in {elapsed:.2f}s. Trying next fallback model."
                            )
                            break

                    logger.error(f"Stream Gemini '{model_name}' returned unretryable status {status} in {elapsed:.2f}s: {body_text[:200]}")
                    last_error = RuntimeError(f"AI service error ({status}).")
                    break

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                elapsed = time.time() - start_time
                reason = type(exc).__name__

                # Task 3: If tokens already sent to user, DO NOT retry (avoid duplicate output)
                if tokens_yielded > 0:
                    logger.error(
                        f"Stream connection interrupted mid-stream for '{model_name}' after {tokens_yielded} tokens ({reason}) in {elapsed:.2f}s."
                    )
                    raise RuntimeError("Streaming connection interrupted mid-stream.") from exc

                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    base_delay = BACKOFF_DELAYS[attempt] if attempt < len(BACKOFF_DELAYS) else 2.0
                    delay = base_delay + random.uniform(0.1, 0.4)
                    logger.warning(
                        f"Stream Network/Timeout on '{model_name}' attempt {attempt + 1}/{MAX_RETRIES} ({reason}) in {elapsed:.2f}s. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Stream retries exhausted for '{model_name}' ({reason}) in {elapsed:.2f}s. Trying next fallback model."
                    )
                    break

    # Task 4: Saare streaming attempts fail hon to ek baar non-streaming generateContent try karo
    if tokens_yielded == 0:
        logger.info("All streaming attempts failed; attempting non-streaming fallback (generateContent)...")
        try:
            fallback_text = ask(system, messages, max_tokens=max_tokens)
            if fallback_text:
                yield fallback_text
                return
        except Exception as non_stream_exc:
            logger.warning(f"Non-streaming fallback also failed: {type(non_stream_exc).__name__}")

    if last_error:
        raise last_error
    raise RuntimeError("AI service temporarily unavailable. Please try again later.")
