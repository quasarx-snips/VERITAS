"""Gemini text API adapter for VERITAS's zero-pixel explanation contract."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from veritas.runtime_config import GeminiSettings


@dataclass(frozen=True)
class LLMRuntimeResult:
    provider: str
    model: str
    success: bool
    text: str = ""
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "success": self.success,
                "text": self.text, "latency_ms": self.latency_ms, "error": self.error}


class GeminiClient:
    """Minimal REST client; it avoids an additional SDK dependency."""

    def __init__(self, settings: GeminiSettings | None = None) -> None:
        self.settings = settings or GeminiSettings.from_env()

    @property
    def configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.model)

    def complete(self, system: str, user: str) -> LLMRuntimeResult:
        if not self.configured:
            return LLMRuntimeResult("gemini", self.settings.model, False, error="Gemini API key is not configured")
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": self.settings.temperature,
                                     "maxOutputTokens": self.settings.max_tokens}}
        endpoint = ("https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{quote(self.settings.model, safe='')}:generateContent?key={quote(self.settings.api_key, safe='')}")
        request = Request(endpoint, data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # nosec B310: fixed Gemini API host
                payload: Mapping[str, Any] = json.loads(response.read().decode("utf-8"))
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, Mapping))
            if not text.strip():
                raise ValueError("Gemini returned an empty or malformed completion")
            return LLMRuntimeResult("gemini", self.settings.model, True, text.strip(),
                                    round((time.perf_counter() - started) * 1000, 2))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return LLMRuntimeResult("gemini", self.settings.model, False, latency_ms=round((time.perf_counter() - started) * 1000, 2), error=str(exc))

    def chat(self, system: str, messages: list[dict[str, str]]) -> LLMRuntimeResult:
        """Multi-turn chat completion using conversation history."""
        if not self.configured:
            return LLMRuntimeResult("gemini", self.settings.model, False, error="Gemini API key is not configured")
        contents = []
        for msg in messages:
            role = "user" if msg.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": self.settings.temperature,
                "maxOutputTokens": self.settings.max_tokens,
            },
        }
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.settings.model, safe='')}:generateContent?key={quote(self.settings.api_key, safe='')}"
        )
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # nosec B310
                payload: Mapping[str, Any] = json.loads(response.read().decode("utf-8"))
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, Mapping))
            if not text.strip():
                raise ValueError("Gemini returned an empty or malformed completion")
            return LLMRuntimeResult("gemini", self.settings.model, True, text.strip(),
                                    round((time.perf_counter() - started) * 1000, 2))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return LLMRuntimeResult("gemini", self.settings.model, False, latency_ms=round((time.perf_counter() - started) * 1000, 2), error=str(exc))

