"""
Groq LLM Provider Module
Exclusively uses Groq (via LangChain ChatGroq) for all text generation, chat streaming, and JSON extraction.
Zero Gemini LLM dependencies.
"""
import hashlib
import json
import logging
import os
import random
import re
import time
from typing import Any, Generator, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Suppress verbose HTTP/transport logging to prevent sensitive header/URL leaks
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_groq").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("langsmith").setLevel(logging.WARNING)

logger = logging.getLogger("llm")

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# Configuration & In-Memory Response Cache
# ---------------------------------------------------------------------------
MAX_RETRIES_PER_MODEL = 2
RETRY_BACKOFF_DELAYS = [0.5, 1.0]

_response_cache: dict = {}  # {cache_key: (response_text, timestamp)}
_CACHE_TTL = 600            # 10 minutes


def _cache_key(system: str, messages: list) -> str:
    raw = system + json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    entry = _response_cache.get(key)
    if entry and (time.time() - entry[1]) < _CACHE_TTL:
        return entry[0]
    return None


def _cache_set(key: str, value: str) -> None:
    _response_cache[key] = (value, time.time())


def get_configured_models() -> List[str]:
    """Returns primary and fallback Groq models from environment variables."""
    primary = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    fallback = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant").strip()

    models = [primary]
    if fallback and fallback not in models:
        models.append(fallback)
    return models


def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY is not set in environment.")
    return key


def _create_groq_client(
    model_name: str,
    temperature: float = 0.7,
    timeout: float = 30.0,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> ChatGroq:
    """Creates a ChatGroq client instance with configured parameters."""
    api_key = _get_api_key()
    kwargs: dict = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        max_retries=1,  # We manage application-level retries across fallback models explicitly
        api_key=api_key,
        **kwargs,
    )


def _build_lc_messages(system: str, messages: list) -> list:
    """Converts message dicts to LangChain message objects."""
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


# ---------------------------------------------------------------------------
# Public API: ask()
# ---------------------------------------------------------------------------
def ask(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
    deterministic: bool = False,
    json_mode: bool = False,
) -> str:
    """
    Non-streaming generation using Groq.
    Tries primary model first, with up to 2 retries on transient errors,
    then fails over to fallback model.
    """
    # 1. Cache lookup (skip for json_mode to ensure freshest schema enforcement)
    if not json_mode:
        ck = _cache_key(system, messages)
        cached = _cache_get(ck)
        if cached is not None:
            logger.info("ask(): Returning cached response.")
            return cached

    candidate_models = get_configured_models()
    last_error: Optional[Exception] = None
    temperature = 0.0 if deterministic else 0.7
    timeout = 30.0

    for model_name in candidate_models:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            start_time = time.time()
            try:
                lc_messages = _build_lc_messages(system, messages)

                # Task 3: If Groq supports it, enable JSON mode; if not supported, fall back silently
                try:
                    client = _create_groq_client(
                        model_name,
                        temperature=temperature,
                        timeout=timeout,
                        json_mode=json_mode,
                        max_tokens=max_tokens,
                    )
                    response = client.invoke(lc_messages)
                except Exception as model_err:
                    err_str = str(model_err).lower()
                    if json_mode and any(k in err_str for k in ["response_format", "json_object", "unsupported", "400"]):
                        logger.warning(
                            f"Model '{model_name}' does not support response_format json_object ({model_err}); falling back to standard call."
                        )
                        client = _create_groq_client(
                            model_name,
                            temperature=temperature,
                            timeout=timeout,
                            json_mode=False,
                            max_tokens=max_tokens,
                        )
                        response = client.invoke(lc_messages)
                    else:
                        raise

                elapsed = time.time() - start_time
                text = (response.content or "").strip()

                # Task 3: If model returns empty content (reasoning models can use up all tokens), retry once with larger max_tokens
                if not text:
                    logger.warning(
                        f"Groq '{model_name}' returned empty content (possibly reasoning token exhaustion). Retrying with larger max_tokens..."
                    )
                    boosted_tokens = (max_tokens or 2000) * 2
                    retry_client = _create_groq_client(
                        model_name,
                        temperature=temperature,
                        timeout=timeout,
                        json_mode=json_mode,
                        max_tokens=boosted_tokens,
                    )
                    try:
                        response = retry_client.invoke(lc_messages)
                        text = (response.content or "").strip()
                    except Exception as retry_err:
                        logger.warning(f"Retry with larger max_tokens failed: {retry_err}")

                logger.info(
                    f"Groq '{model_name}' success in {elapsed:.2f}s (attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL})"
                )
                if not json_mode and text:
                    _cache_set(ck, text)
                return text

            except Exception as exc:
                elapsed = time.time() - start_time
                reason = type(exc).__name__
                last_error = exc
                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    base_delay = RETRY_BACKOFF_DELAYS[attempt] if attempt < len(RETRY_BACKOFF_DELAYS) else 1.0
                    delay = base_delay + random.uniform(0.05, 0.2)
                    logger.warning(
                        f"Groq '{model_name}' attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL} failed ({reason}) in {elapsed:.2f}s. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Groq '{model_name}' exhausted {MAX_RETRIES_PER_MODEL} attempts ({reason}) in {elapsed:.2f}s. Trying fallback model..."
                    )
                    break

    # If all models failed, raise one clear error including the underlying error
    raise RuntimeError(
        f"Groq service unavailable across all models ({candidate_models}). Last error: {type(last_error).__name__}: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API: ask_stream()
# ---------------------------------------------------------------------------
def ask_stream(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
) -> Generator[str, None, None]:
    """
    Streaming generator using Groq.
    Streams token chunks directly from ChatGroq.stream().
    If an error occurs before any tokens are yielded, retries or falls back to next model.
    If an error occurs after tokens have already been yielded, terminates immediately without retry
    to prevent duplicate responses in the user UI.
    """
    candidate_models = get_configured_models()
    tokens_yielded = 0
    last_error: Optional[Exception] = None

    for model_name in candidate_models:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            start_time = time.time()
            try:
                client = _create_groq_client(model_name, temperature=0.7, timeout=30.0)
                lc_messages = _build_lc_messages(system, messages)

                for chunk in client.stream(lc_messages):
                    txt = chunk.content if hasattr(chunk, "content") else ""
                    if txt:
                        tokens_yielded += 1
                        yield txt

                elapsed = time.time() - start_time
                logger.info(
                    f"Groq '{model_name}' stream complete, {tokens_yielded} tokens in {elapsed:.2f}s (attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL})"
                )
                return

            except Exception as exc:
                elapsed = time.time() - start_time
                reason = type(exc).__name__

                # If tokens have already been yielded to the user, DO NOT retry or switch models
                if tokens_yielded > 0:
                    logger.error(
                        f"Groq '{model_name}' stream interrupted mid-stream after {tokens_yielded} tokens ({reason}) in {elapsed:.2f}s."
                    )
                    raise RuntimeError("Streaming connection interrupted mid-stream.") from exc

                last_error = exc
                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    base_delay = RETRY_BACKOFF_DELAYS[attempt] if attempt < len(RETRY_BACKOFF_DELAYS) else 1.0
                    delay = base_delay + random.uniform(0.05, 0.2)
                    logger.warning(
                        f"Groq stream '{model_name}' attempt {attempt + 1}/{MAX_RETRIES_PER_MODEL} failed ({reason}) in {elapsed:.2f}s. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Groq stream retries exhausted for '{model_name}' ({reason}) in {elapsed:.2f}s. Trying fallback model..."
                    )
                    break

    raise RuntimeError(
        f"Groq streaming unavailable across all models ({candidate_models}). Last error: {type(last_error).__name__}: {last_error}"
    )


# ---------------------------------------------------------------------------
# Public API: ask_json()
# ---------------------------------------------------------------------------
def _clean_and_parse_json(raw: str) -> Any:
    """Helper to strip markdown code fences, clean trailing commas, and extract valid JSON."""
    raw = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Match JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
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


def ask_json(system: str, user: str, max_tokens: int | None = None) -> Any:
    """
    JSON extraction using Groq with markdown code fence stripping and automatic retry.
    """
    effective_tokens = max_tokens or 3000
    try:
        raw = ask(
            system,
            [{"role": "user", "content": user}],
            max_tokens=effective_tokens,
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
            max_tokens=max(effective_tokens, 4000),
            deterministic=True,
            json_mode=True,
        )
        return _clean_and_parse_json(raw)
