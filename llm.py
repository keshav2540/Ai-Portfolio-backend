import json
import os
import re
from typing import Generator
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

raw_model = (os.getenv("MODEL") or "").strip()
if not raw_model or not ("gemini" in raw_model.lower() or "gemma" in raw_model.lower()):
    MODEL = "gemini-3.5-flash"
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


def ask(
    system: str,
    messages: list[dict],
    max_tokens: int | None = None,
    deterministic: bool = False,
    json_mode: bool = False,
) -> str:
    key = _get_api_key()
    url = f"{BASE_URL}/models/{MODEL}:generateContent?key={key}"

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

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {resp.text}")
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        return "".join(text_parts).strip()


def ask_json(system: str, user: str, max_tokens: int | None = None) -> dict:
    raw = ask(
        system,
        [{"role": "user", "content": user}],
        max_tokens,
        deterministic=True,
        json_mode=True,
    )
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    return json.loads(raw)


def ask_stream(system: str, messages: list[dict], max_tokens: int | None = None) -> Generator[str, None, None]:
    key = _get_api_key()
    url = f"{BASE_URL}/models/{MODEL}:streamGenerateContent?alt=sse&key={key}"

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

    with httpx.Client(timeout=30.0) as client:
        with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini Streaming error ({resp.status_code}): {resp.text}")
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
