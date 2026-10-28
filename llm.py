import json
import logging
import os
import re
from typing import Generator
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.8-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

raw_model = (os.getenv("MODEL") or "").strip()
# Detect missing, empty, or older deprecated model names
if not raw_model or raw_model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-3.5-flash"] or not ("gemini" in raw_model.lower() or "gemma" in raw_model.lower()):
    MODEL = DEFAULT_MODEL
else:
    MODEL = raw_model


def _get_api_key() -> str:
    key = os.getenv("gemini_api_key") or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("gemini_api_key is not set in environment.")
    return key


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


def _candidate_models() -> list[str]:
    global MODEL
    models = [MODEL]
    for fb in FALLBACK_MODELS:
        if fb not in models:
            models.append(fb)
    return models


def ask(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
    deterministic: bool = False,
    json_mode: bool = False,
) -> str:
    global MODEL
    key = _get_api_key()

    payload = {
        "contents": _format_contents(messages),
    }
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

    models_to_try = _candidate_models()
    last_error = None

    with httpx.Client(timeout=60.0) as client:
        for model_name in models_to_try:
            url = f"{BASE_URL}/models/{model_name}:generateContent?key={key}"
            try:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    # Remember working model
                    if model_name != MODEL:
                        logger.info(f"Switched active model to working fallback: {model_name}")
                        MODEL = model_name
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        return ""
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_parts = [p.get("text", "") for p in parts if "text" in p]
                    return "".join(text_parts).strip()
                elif resp.status_code == 404 or "not found" in resp.text.lower():
                    logger.warning(f"Model '{model_name}' not found (404). Trying next fallback...")
                    last_error = RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
                    continue
                else:
                    raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.error(f"Network error contacting Gemini ({model_name}): {e}")
                last_error = e

    if last_error:
        raise last_error
    raise RuntimeError("Failed to get response from Gemini AI models.")


def ask_json(system: str, user: str, max_tokens: int | None = None) -> dict:
    raw = ask(
        system,
        [{"role": "user", "content": user}],
        max_tokens,
        deterministic=True,
        json_mode=True,
    )
    raw = raw.strip()

    # 1. Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass

    # 2. Extract outermost JSON object { ... }
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Clean trailing commas and retry
    try:
        candidate = re.sub(r",\s*([\]}])", r"\1", match.group(1) if match else cleaned)
        return json.loads(candidate)
    except Exception as e:
        logger.error(f"JSON parsing failed on raw text:\n{raw[:300]}")
        raise ValueError(f"Could not parse valid JSON from AI response: {e}")


def ask_stream(system: str, messages: list[dict], max_tokens: int | None = None) -> Generator[str, None, None]:
    global MODEL
    key = _get_api_key()

    payload = {
        "contents": _format_contents(messages),
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    gen_cfg = {}
    if max_tokens:
        gen_cfg["maxOutputTokens"] = max_tokens
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    models_to_try = _candidate_models()
    client = httpx.Client(timeout=60.0)

    try:
        for model_name in models_to_try:
            url = f"{BASE_URL}/models/{model_name}:streamGenerateContent?alt=sse&key={key}"
            try:
                with client.stream("POST", url, json=payload) as resp:
                    if resp.status_code == 200:
                        if model_name != MODEL:
                            logger.info(f"Stream switched active model to: {model_name}")
                            MODEL = model_name
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
                                                if "text" in p and p["text"]:
                                                    yield p["text"]
                                    except Exception:
                                        continue
                        return
                    elif resp.status_code == 404 or "not found" in resp.text.lower():
                        logger.warning(f"Streaming model '{model_name}' not found. Trying next fallback...")
                        continue
                    else:
                        raise RuntimeError(f"Gemini Streaming error ({resp.status_code}): {resp.text}")
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.error(f"Network error in ask_stream with {model_name}: {e}")
                continue
    finally:
        client.close()
